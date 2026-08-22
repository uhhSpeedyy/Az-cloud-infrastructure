terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# Resource Group
resource "azurerm_resource_group" "resourcegroup" {
  name     = "resourcegroup1"
  location = "Australia East"
}

# Virtual Network
resource "azurerm_virtual_network" "vnet" {
  name                = "vnet1"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.resourcegroup.location
  resource_group_name = azurerm_resource_group.resourcegroup.name
}


#Subnet
resource "azurerm_subnet" "subnet_one" {
  name                 = "subnet-one"
  resource_group_name  = azurerm_resource_group.resourcegroup.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.1.0/24"]
}

#Application Subnet
resource "azurerm_subnet" "app_subnet" {
  name                 = "app-subnet"
  resource_group_name  = azurerm_resource_group.resourcegroup.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.2.0/24"]

  #delegration block
  delegation {
    name = "app-service-delegation"

    service_delegation {
      name = "Microsoft.Web/serverFarms"

      actions = [
        "Microsoft.Network/virtualNetworks/subnets/action"
      ]
    }
  }
}

# App Service Plan
resource "azurerm_service_plan" "app_plan" {
  name                = "app-service-plan"
  resource_group_name = azurerm_resource_group.resourcegroup.name
  location            = azurerm_resource_group.resourcegroup.location
  os_type             = "Linux"
  sku_name            = "B1"
}

# Network Security Group (NSG)
resource "azurerm_network_security_group" "subnet_one_nsg" {
  name                = "subnet-one-nsg"
  location            = azurerm_resource_group.resourcegroup.location
  resource_group_name = azurerm_resource_group.resourcegroup.name
}

resource "azurerm_subnet_network_security_group_association" "subnet_one_nsg_association" {
  subnet_id                 = azurerm_subnet.subnet_one.id
  network_security_group_id = azurerm_network_security_group.subnet_one_nsg.id
}


#Azure SQL Server
variable "sql_server_resource_id" {
  description = "Resource ID of the existing Azure SQL Server"
  type        = string
}

variable "sql_server_fqdn" {
  description = "Fully qualified domain name of the existing Azure SQL Server"
  type        = string
  default     = "speedserver.database.windows.net"
}

variable "sql_database_name" {
  description = "Azure SQL database used by the AFL prediction application"
  type        = string
  default     = "DB_one"
}

variable "squiggle_contact" {
  description = "Contact identifier included in the Squiggle API User-Agent"
  type        = string
  default     = "github.com/uhhSpeedyy/Az-cloud-infrastructure"
}

variable "afl_refresh_token" {
  description = "Shared secret used by the scheduled prediction refresh endpoint"
  type        = string
  sensitive   = true
  default     = null
}


#Private Endpoint for Azure SQL Server
resource "azurerm_private_endpoint" "sql_pe" {
  name                = "sql-private-endpoint"
  location            = azurerm_resource_group.resourcegroup.location
  resource_group_name = azurerm_resource_group.resourcegroup.name
  subnet_id           = azurerm_subnet.subnet_one.id

  private_service_connection {
    name                           = "sql-psc"
    is_manual_connection           = false
    private_connection_resource_id = var.sql_server_resource_id
    subresource_names              = ["sqlServer"]
  }

  private_dns_zone_group {
    name                 = "sql-dns-zone-group"
    private_dns_zone_ids = [azurerm_private_dns_zone.sql_privatedns.id]
  }
}

#Private DNS Zone for Azure SQL Server
resource "azurerm_private_dns_zone" "sql_privatedns" {
  name                = "privatelink.database.windows.net"
  resource_group_name = azurerm_resource_group.resourcegroup.name
}


#DNS Zone to VNET Link
resource "azurerm_private_dns_zone_virtual_network_link" "vnet_link" {
  name                  = "vnet-link-to-privatelink-sql"
  resource_group_name   = azurerm_resource_group.resourcegroup.name
  private_dns_zone_name = azurerm_private_dns_zone.sql_privatedns.name
  virtual_network_id    = azurerm_virtual_network.vnet.id
  registration_enabled  = false
}


#app service
resource "azurerm_linux_web_app" "app" {
  name                = "Sam-Speed"
  resource_group_name = azurerm_resource_group.resourcegroup.name
  location            = azurerm_resource_group.resourcegroup.location
  service_plan_id     = azurerm_service_plan.app_plan.id
  identity {
    type = "SystemAssigned"
  }
  https_only = true

  site_config {
    # Keep this off so App Service does not ping `/` every five minutes and
    # repeatedly wake the serverless Azure SQL database while the site is idle.
    always_on         = false
    health_check_path = "/health"
    app_command_line  = "gunicorn --bind=0.0.0.0:8000 --workers=1 --threads=4 --timeout=230 app:app"

    application_stack {
      python_version = "3.11"
    }
  }

  app_settings = merge({
    DB_SERVER                      = var.sql_server_fqdn
    DB_NAME                        = var.sql_database_name
    AFL_DATABASE_ENABLED           = "true"
    AFL_DATABASE_READ_ENABLED      = "false"
    AFL_HOLDOUT_SEASON             = "2022"
    AFL_START_SEASON               = "2012"
    AFL_CURRENT_SEASON             = "2026"
    SQUIGGLE_CONTACT               = var.squiggle_contact
    SCM_DO_BUILD_DURING_DEPLOYMENT = "true"
    ENABLE_ORYX_BUILD              = "true"
    }, var.afl_refresh_token == null ? {} : {
    AFL_REFRESH_TOKEN = var.afl_refresh_token
  })
}

#app service VNS
resource "azurerm_app_service_virtual_network_swift_connection" "app_vnet" {
  app_service_id = azurerm_linux_web_app.app.id
  subnet_id      = azurerm_subnet.app_subnet.id
}

output "web_app_url" {
  value = "https://${azurerm_linux_web_app.app.default_hostname}"
}

output "web_app_managed_identity_principal_id" {
  value = azurerm_linux_web_app.app.identity[0].principal_id
}
