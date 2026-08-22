(() => {
  "use strict";

  const STORAGE_KEY = "sam-speed-reading-room-v1";
  const MAX_FAVOURITES = 10;
  const SEARCH_DELAY_MS = 320;

  const form = document.getElementById("recommendation-form");
  const searchInput = document.getElementById("book-search");
  const suggestionsElement = document.getElementById("book-suggestions");
  const searchProgress = document.getElementById("search-progress");
  const searchStatus = document.getElementById("search-status");
  const favouritesGrid = document.getElementById("favourites-grid");
  const selectionCount = document.getElementById("selection-count");
  const clearButton = document.getElementById("clear-favourites");
  const recommendButton = document.getElementById("recommend-button");
  const loadingPanel = document.getElementById("loading-panel");
  const errorPanel = document.getElementById("error-panel");
  const errorMessage = document.getElementById("error-message");
  const retryButton = document.getElementById("retry-button");
  const results = document.getElementById("results");
  const profileSummary = document.getElementById("profile-summary");
  const profileThemesWrap = document.getElementById("profile-themes-wrap");
  const profileThemes = document.getElementById("profile-themes");
  const profileStylesWrap = document.getElementById("profile-styles-wrap");
  const profileStyles = document.getElementById("profile-styles");
  const resultLists = document.getElementById("result-lists");

  if (!form || !searchInput || !suggestionsElement) return;

  const state = {
    favourites: [],
    suggestions: [],
    activeSuggestion: -1,
    searchTimer: null,
    searchController: null,
    searchRequest: 0,
    recommendationController: null,
    isLoading: false,
  };

  function cleanText(value, fallback = "") {
    if (value === null || value === undefined) return fallback;
    return String(value).trim() || fallback;
  }

  function cleanList(value) {
    const values = Array.isArray(value) ? value : value ? [value] : [];
    return values
      .map((item) => {
        if (typeof item === "string" || typeof item === "number") return cleanText(item);
        if (item && typeof item === "object") {
          return cleanText(item.name || item.label || item.title || item.description);
        }
        return "";
      })
      .filter(Boolean);
  }

  function usesOnlyLatinLetters(value) {
    const letters = cleanText(value).match(/\p{Letter}/gu) || [];
    return letters.length > 0 && letters.every((letter) => /\p{Script=Latin}/u.test(letter));
  }

  function normaliseBook(raw = {}) {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;

    const languages = cleanList(raw.language);
    const rawAuthors = cleanList(raw.authors || raw.author_name || raw.author);
    const hasEnglishEdition = !languages.length || languages.some((language) => {
      const value = language.toLowerCase();
      return value === "en" || value === "eng" || value === "english" || value.includes("/eng");
    });
    const latinAuthors = rawAuthors.filter(usesOnlyLatinLetters);
    const authors = hasEnglishEdition && latinAuthors.length ? latinAuthors : rawAuthors;
    const yearValue = raw.year ?? raw.first_publish_year ?? raw.publish_year;
    const parsedYear = Number.parseInt(yearValue, 10);
    const numberOrNull = (value, integer = true) => {
      if (value === null || value === undefined || value === "") return null;
      const parsed = integer ? Number.parseInt(value, 10) : Number.parseFloat(value);
      return Number.isFinite(parsed) ? parsed : null;
    };
    const year = Number.isFinite(parsedYear) ? parsedYear : null;
    const subjects = cleanList(raw.subject || raw.subjects);
    const themes = cleanList(raw.themes || raw.matched_themes);

    return {
      key: cleanText(raw.key || raw.work_key || raw.id),
      title: cleanText(raw.title, "Untitled book"),
      authors,
      year,
      cover_url: cleanText(raw.cover_url || raw.cover),
      open_library_url: cleanText(raw.open_library_url || raw.url),
      length_label: cleanText(raw.length_label),
      themes,
      reasons: cleanList(raw.reasons || raw.reason),
      // Preserve the Open Library fields used by the ranking model while also
      // exposing compact aliases above for the browser UI.
      author_name: authors,
      author_key: cleanList(raw.author_key),
      first_publish_year: year,
      number_of_pages_median: numberOrNull(raw.number_of_pages_median),
      subject: subjects,
      language: languages,
      first_sentence: cleanList(raw.first_sentence),
      cover_i: numberOrNull(raw.cover_i),
      edition_count: numberOrNull(raw.edition_count) || 0,
      ratings_average: numberOrNull(raw.ratings_average, false),
      ratings_count: numberOrNull(raw.ratings_count) || 0,
      readinglog_count: numberOrNull(raw.readinglog_count) || 0,
      want_to_read_count: numberOrNull(raw.want_to_read_count) || 0,
      currently_reading_count: numberOrNull(raw.currently_reading_count) || 0,
      already_read_count: numberOrNull(raw.already_read_count) || 0,
    };
  }

  function bookIdentity(book) {
    if (book.key) return book.key.toLowerCase();
    return `${book.title}|${book.authors.join("|")}`.toLowerCase();
  }

  function authorLine(book) {
    return book.authors.length ? book.authors.join(", ") : "Unknown author";
  }

  function safeUrl(value, fallback = "", allowedHostname = "") {
    const candidate = cleanText(value, fallback);
    if (!candidate) return "";

    try {
      const parsed = new URL(candidate, window.location.origin);
      if (
        parsed.protocol === "https:"
        && (!allowedHostname || parsed.hostname === allowedHostname)
      ) return parsed.href;
    } catch (_error) {
      return "";
    }
    return "";
  }

  function openLibraryUrl(book) {
    const fallback = book.key.startsWith("/works/")
      ? `https://openlibrary.org${book.key}`
      : "https://openlibrary.org/";
    return safeUrl(book.open_library_url, fallback, "openlibrary.org");
  }

  function appendCover(container, book, options = {}) {
    container.setAttribute("aria-hidden", "true");
    const fallbackText = cleanText(options.fallbackText, book.title.slice(0, 1).toUpperCase() || "B");
    const imageUrl = safeUrl(book.cover_url, "", "covers.openlibrary.org");

    if (!imageUrl) {
      container.textContent = fallbackText;
      if (options.fallbackClass) container.classList.add(options.fallbackClass);
      return;
    }

    const image = document.createElement("img");
    image.src = imageUrl;
    image.alt = "";
    image.decoding = "async";
    if (options.lazy) image.loading = "lazy";
    image.addEventListener("error", () => {
      container.replaceChildren(document.createTextNode(fallbackText));
      if (options.fallbackClass) container.classList.add(options.fallbackClass);
    }, { once: true });
    container.append(image);
  }

  function announceSearch(message, isError = false) {
    searchStatus.textContent = message;
    searchStatus.style.color = isError ? "var(--error)" : "";
  }

  function setSearchBusy(isBusy) {
    searchProgress.classList.toggle("is-active", isBusy);
    searchInput.setAttribute("aria-busy", String(isBusy));
  }

  function closeSuggestions() {
    suggestionsElement.hidden = true;
    searchInput.setAttribute("aria-expanded", "false");
    searchInput.removeAttribute("aria-activedescendant");
    state.activeSuggestion = -1;
    suggestionsElement.querySelectorAll(".suggestion-option").forEach((option) => {
      option.classList.remove("is-active");
      option.setAttribute("aria-selected", "false");
    });
  }

  function openSuggestions() {
    suggestionsElement.hidden = false;
    searchInput.setAttribute("aria-expanded", "true");
  }

  function setActiveSuggestion(index) {
    const options = [...suggestionsElement.querySelectorAll(".suggestion-option")];
    if (!options.length) {
      state.activeSuggestion = -1;
      searchInput.removeAttribute("aria-activedescendant");
      return;
    }

    const nextIndex = ((index % options.length) + options.length) % options.length;
    state.activeSuggestion = nextIndex;

    options.forEach((option, optionIndex) => {
      const isActive = optionIndex === nextIndex;
      option.classList.toggle("is-active", isActive);
      option.setAttribute("aria-selected", String(isActive));
    });

    const activeOption = options[nextIndex];
    searchInput.setAttribute("aria-activedescendant", activeOption.id);
    activeOption.scrollIntoView({ block: "nearest" });
  }

  function renderSuggestions(books) {
    suggestionsElement.replaceChildren();
    state.suggestions = books;
    state.activeSuggestion = -1;

    if (!books.length) {
      closeSuggestions();
      return;
    }

    books.forEach((book, index) => {
      const option = document.createElement("button");
      option.type = "button";
      option.className = "suggestion-option";
      option.id = `book-suggestion-${index}`;
      option.setAttribute("role", "option");
      option.setAttribute("aria-selected", "false");
      option.setAttribute("aria-posinset", String(index + 1));
      option.setAttribute("aria-setsize", String(books.length));
      option.tabIndex = -1;

      const cover = document.createElement("span");
      cover.className = "suggestion-cover";
      appendCover(cover, book);

      const copy = document.createElement("span");
      copy.className = "suggestion-copy";
      const title = document.createElement("strong");
      title.textContent = book.title;
      const author = document.createElement("span");
      author.textContent = authorLine(book);
      copy.append(title, author);

      const year = document.createElement("span");
      year.className = "suggestion-year";
      year.textContent = book.year ? String(book.year) : "Year unknown";

      option.append(cover, copy, year);
      option.addEventListener("pointerenter", () => setActiveSuggestion(index));
      option.addEventListener("mousedown", (event) => event.preventDefault());
      option.addEventListener("click", () => addFavourite(book));
      suggestionsElement.append(option);
    });

    openSuggestions();
  }

  async function searchBooks(query) {
    if (state.searchController) state.searchController.abort();
    state.searchController = new AbortController();
    const requestNumber = ++state.searchRequest;
    setSearchBusy(true);
    announceSearch("");

    try {
      const parameters = new URLSearchParams({ q: query });
      const response = await fetch(`/api/books/search?${parameters.toString()}`, {
        headers: { Accept: "application/json" },
        signal: state.searchController.signal,
      });

      if (!response.ok) throw new Error(`search:${response.status}`);
      const payload = await response.json();
      if (requestNumber !== state.searchRequest || searchInput.value.trim() !== query) return;

      const selected = new Set(state.favourites.map(bookIdentity));
      const books = (Array.isArray(payload.books) ? payload.books : [])
        .map(normaliseBook)
        .filter((book) => book && book.key && book.title !== "Untitled book")
        .filter((book) => !selected.has(bookIdentity(book)))
        .slice(0, 10);

      renderSuggestions(books);
      if (!books.length) announceSearch("No matching books found.");
    } catch (error) {
      if (error.name === "AbortError") return;
      if (requestNumber !== state.searchRequest) return;
      state.suggestions = [];
      suggestionsElement.replaceChildren();
      closeSuggestions();
      const rateLimited = String(error.message).includes("search:429");
      announceSearch(
        rateLimited
          ? "Search is busy just now. Please wait a moment and try again."
          : "Book search is temporarily unavailable. Please try again.",
        true,
      );
    } finally {
      if (requestNumber === state.searchRequest) setSearchBusy(false);
    }
  }

  function scheduleSearch() {
    window.clearTimeout(state.searchTimer);
    if (state.searchController) state.searchController.abort();
    state.searchRequest += 1;
    setSearchBusy(false);

    const query = searchInput.value.trim();
    announceSearch("");
    state.suggestions = [];
    suggestionsElement.replaceChildren();
    closeSuggestions();
    if (query.length < 2) {
      return;
    }

    state.searchTimer = window.setTimeout(() => searchBooks(query), SEARCH_DELAY_MS);
  }

  function getPreferences() {
    const valueFor = (name, fallback) => form.querySelector(`input[name="${name}"]:checked`)?.value || fallback;
    return {
      era: valueFor("era", "balanced"),
      length: valueFor("length", "similar"),
      discovery: valueFor("discovery", "balanced"),
    };
  }

  function saveState() {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify({
        favourites: state.favourites,
        preferences: getPreferences(),
      }));
    } catch (_error) {
      // Storage can be unavailable in private or tightly restricted browser modes.
    }
  }

  function restoreState() {
    try {
      const stored = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "{}");
      const seen = new Set();
      state.favourites = (Array.isArray(stored.favourites) ? stored.favourites : [])
        .map(normaliseBook)
        .filter((book) => {
          if (!book || !book.key || book.title === "Untitled book") return false;
          const identity = bookIdentity(book);
          if (!identity || seen.has(identity)) return false;
          seen.add(identity);
          return true;
        })
        .slice(0, MAX_FAVOURITES);

      const allowed = {
        era: ["balanced", "modern", "classics"],
        length: ["similar", "any"],
        discovery: ["balanced", "familiar", "adventurous"],
      };
      const preferences = stored.preferences || {};
      Object.entries(allowed).forEach(([name, values]) => {
        const value = values.includes(preferences[name]) ? preferences[name] : null;
        if (value) {
          const input = form.querySelector(`input[name="${name}"][value="${value}"]`);
          if (input) input.checked = true;
        }
      });
    } catch (_error) {
      state.favourites = [];
    }
  }

  function invalidateResults() {
    if (state.recommendationController) state.recommendationController.abort();
    state.isLoading = false;
    loadingPanel.hidden = true;
    errorPanel.hidden = true;
    results.hidden = true;
  }

  function renderFavourites() {
    favouritesGrid.replaceChildren();

    if (!state.favourites.length) {
      const empty = document.createElement("div");
      empty.className = "empty-shelf";
      empty.id = "empty-shelf";
      const message = document.createElement("p");
      message.textContent = "No books selected.";
      empty.append(message);
      favouritesGrid.append(empty);
    } else {
      state.favourites.forEach((book) => {
        const card = document.createElement("article");
        card.className = "favourite-card";

        const cover = document.createElement("div");
        cover.className = "favourite-cover";
        appendCover(cover, book);

        const copy = document.createElement("div");
        copy.className = "favourite-copy";
        const title = document.createElement("strong");
        title.textContent = book.title;
        const author = document.createElement("span");
        author.textContent = authorLine(book);
        copy.append(title, author);

        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "remove-favourite";
        remove.setAttribute("aria-label", `Remove ${book.title} from favourites`);
        remove.textContent = "×";
        remove.addEventListener("click", () => removeFavourite(book));

        card.append(cover, copy, remove);
        favouritesGrid.append(card);
      });
    }

    selectionCount.replaceChildren();
    const count = document.createElement("strong");
    count.textContent = String(state.favourites.length);
    selectionCount.append(count, document.createTextNode(" / 10 selected"));
    clearButton.hidden = state.favourites.length === 0;
    recommendButton.disabled = state.favourites.length === 0 || state.isLoading;
  }

  function addFavourite(book) {
    const normalizedBook = normaliseBook(book);
    if (!normalizedBook || !normalizedBook.key || normalizedBook.title === "Untitled book") {
      announceSearch("That book could not be added. Please choose another result.", true);
      closeSuggestions();
      return;
    }

    if (state.favourites.length >= MAX_FAVOURITES) {
      announceSearch("Your shelf is full. Remove a book before adding another.", true);
      closeSuggestions();
      return;
    }

    if (state.favourites.some((favourite) => bookIdentity(favourite) === bookIdentity(normalizedBook))) {
      announceSearch(`${normalizedBook.title} is already on your shelf.`);
      closeSuggestions();
      return;
    }

    state.favourites.push(normalizedBook);
    searchInput.value = "";
    state.suggestions = [];
    closeSuggestions();
    announceSearch(`Added ${normalizedBook.title}.`);
    invalidateResults();
    saveState();
    renderFavourites();
    searchInput.focus();
  }

  function removeFavourite(book) {
    const identity = bookIdentity(book);
    state.favourites = state.favourites.filter((favourite) => bookIdentity(favourite) !== identity);
    invalidateResults();
    saveState();
    renderFavourites();
    announceSearch(`Removed ${book.title}.`);
  }

  function renderSignals(container, wrapper, values) {
    container.replaceChildren();
    const signals = cleanList(values).slice(0, 8);
    wrapper.hidden = signals.length === 0;
    signals.forEach((signal) => {
      const pill = document.createElement("span");
      pill.textContent = signal;
      container.append(pill);
    });
  }

  function createBookCard(rawBook) {
    const book = normaliseBook(rawBook) || normaliseBook({});
    const card = document.createElement("article");
    card.className = "book-card";

    const cover = document.createElement("div");
    cover.className = "book-cover-shell";
    cover.setAttribute("aria-hidden", "true");
    const placeholder = document.createElement("span");
    placeholder.className = "book-cover-placeholder";
    placeholder.textContent = book.title;
    cover.append(placeholder);

    const coverUrl = safeUrl(book.cover_url, "", "covers.openlibrary.org");
    if (coverUrl) {
      const image = document.createElement("img");
      image.src = coverUrl;
      image.alt = "";
      image.loading = "lazy";
      image.decoding = "async";
      image.addEventListener("load", () => placeholder.remove(), { once: true });
      image.addEventListener("error", () => {
        image.remove();
        cover.classList.add("is-fallback");
      }, { once: true });
      cover.prepend(image);
    } else {
      cover.classList.add("is-fallback");
    }

    const body = document.createElement("div");
    body.className = "book-body";
    const title = document.createElement("h4");
    title.textContent = book.title;
    const author = document.createElement("p");
    author.className = "book-author";
    author.textContent = authorLine(book);
    body.append(title, author);

    const metaValues = [book.year ? String(book.year) : "", book.length_label].filter(Boolean);
    if (metaValues.length) {
      const meta = document.createElement("div");
      meta.className = "book-meta";
      metaValues.forEach((value) => {
        const item = document.createElement("span");
        item.textContent = value;
        meta.append(item);
      });
      body.append(meta);
    }

    if (book.themes.length) {
      const themes = document.createElement("div");
      themes.className = "theme-tags";
      book.themes.slice(0, 3).forEach((theme) => {
        const tag = document.createElement("span");
        tag.className = "theme-tag";
        tag.textContent = theme;
        themes.append(tag);
      });
      body.append(themes);
    }

    if (book.reasons.length) {
      const reasons = document.createElement("ul");
      reasons.className = "reason-list";
      book.reasons.slice(0, 3).forEach((reason) => {
        const item = document.createElement("li");
        item.textContent = reason;
        reasons.append(item);
      });
      body.append(reasons);
    }

    const linkUrl = openLibraryUrl(book);
    if (linkUrl) {
      const link = document.createElement("a");
      link.className = "open-library-link";
      link.href = linkUrl;
      link.target = "_blank";
      link.rel = "noreferrer";
      const linkLabel = document.createElement("span");
      linkLabel.textContent = "View on Open Library";
      const arrow = document.createElement("span");
      arrow.setAttribute("aria-hidden", "true");
      arrow.textContent = "↗";
      link.append(linkLabel, arrow);
      body.append(link);
    }

    card.append(cover, body);
    return card;
  }

  function renderResults(payload) {
    const profile = payload && typeof payload.profile === "object" ? payload.profile : {};
    profileSummary.textContent = cleanText(
      profile.summary,
      "Based on your selected books.",
    );
    renderSignals(profileThemes, profileThemesWrap, profile.themes);
    renderSignals(profileStyles, profileStylesWrap, profile.styles);

    resultLists.replaceChildren();
    const lists = Array.isArray(payload?.lists) ? payload.lists : [];
    const populatedLists = lists.filter((list) => Array.isArray(list.books) && list.books.length);

    if (!populatedLists.length) {
      const empty = document.createElement("div");
      empty.className = "no-results";
      const title = document.createElement("h3");
      title.textContent = "No strong matches yet";
      const copy = document.createElement("p");
      copy.textContent = "Add another book or change the options, then try again.";
      empty.append(title, copy);
      resultLists.append(empty);
      return;
    }

    populatedLists.forEach((list, index) => {
      const section = document.createElement("section");
      section.className = "result-list";
      const titleId = `result-list-${index}`;
      section.setAttribute("aria-labelledby", titleId);

      const heading = document.createElement("header");
      heading.className = "list-heading";
      const number = document.createElement("span");
      number.className = "list-index";
      number.setAttribute("aria-hidden", "true");
      number.textContent = String(index + 1).padStart(2, "0");
      const title = document.createElement("h3");
      title.id = titleId;
      title.textContent = cleanText(list.title, "Books to explore");
      const description = document.createElement("p");
      description.textContent = cleanText(list.description, "Matches your selected books.");
      heading.append(number, title, description);

      const grid = document.createElement("div");
      grid.className = "book-grid";
      list.books.forEach((book) => grid.append(createBookCard(book)));

      section.append(heading, grid);
      resultLists.append(section);
    });
  }

  function friendlyRecommendationError(status, payload) {
    const supplied = cleanText(payload?.error || payload?.message);
    if (status === 429) return "The book service is receiving a lot of requests. Wait a moment, then try again.";
    if (status === 502 || status === 503 || status === 504) return "Open Library is temporarily unavailable, so the model cannot gather enough books just now.";
    if (status === 400 && supplied) return supplied;
    return supplied || "Please check your connection and try again in a moment.";
  }

  async function requestRecommendations() {
    if (!state.favourites.length || state.isLoading) return;
    if (state.recommendationController) state.recommendationController.abort();
    state.recommendationController = new AbortController();
    state.isLoading = true;
    renderFavourites();
    results.hidden = true;
    errorPanel.hidden = true;
    loadingPanel.hidden = false;
    loadingPanel.setAttribute("aria-busy", "true");

    try {
      const response = await fetch("/api/books/recommend", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          favourites: state.favourites,
          preferences: getPreferences(),
        }),
        signal: state.recommendationController.signal,
      });

      let payload = {};
      try {
        payload = await response.json();
      } catch (_error) {
        payload = {};
      }

      if (!response.ok) {
        const error = new Error(friendlyRecommendationError(response.status, payload));
        error.isFriendly = true;
        throw error;
      }

      renderResults(payload);
      loadingPanel.hidden = true;
      results.hidden = false;
      window.requestAnimationFrame(() => {
        results.focus({ preventScroll: true });
        results.scrollIntoView({ behavior: preferredScrollBehavior(), block: "start" });
      });
    } catch (error) {
      if (error.name === "AbortError") return;
      loadingPanel.hidden = true;
      results.hidden = true;
      errorMessage.textContent = error.isFriendly
        ? error.message
        : "Please check your connection and try again in a moment.";
      errorPanel.hidden = false;
      errorPanel.scrollIntoView({ behavior: preferredScrollBehavior(), block: "center" });
    } finally {
      loadingPanel.setAttribute("aria-busy", "false");
      state.isLoading = false;
      renderFavourites();
    }
  }

  function preferredScrollBehavior() {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
  }

  searchInput.addEventListener("input", scheduleSearch);
  searchInput.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown" && state.suggestions.length) {
      event.preventDefault();
      openSuggestions();
      setActiveSuggestion(state.activeSuggestion + 1);
    } else if (event.key === "ArrowUp" && state.suggestions.length) {
      event.preventDefault();
      openSuggestions();
      setActiveSuggestion(state.activeSuggestion <= 0 ? state.suggestions.length - 1 : state.activeSuggestion - 1);
    } else if (event.key === "Enter") {
      event.preventDefault();
      if (!suggestionsElement.hidden && state.activeSuggestion >= 0) {
        addFavourite(state.suggestions[state.activeSuggestion]);
      } else if (!suggestionsElement.hidden && state.suggestions.length === 1) {
        addFavourite(state.suggestions[0]);
      } else if (searchInput.value.trim().length >= 2) {
        announceSearch("Choose a book from the suggestions before continuing.");
      }
    } else if (event.key === "Escape") {
      closeSuggestions();
    }
  });

  searchInput.addEventListener("focus", () => {
    if (state.suggestions.length) openSuggestions();
  });

  document.addEventListener("pointerdown", (event) => {
    if (!event.target.closest(".search-wrap")) closeSuggestions();
  });

  clearButton.addEventListener("click", () => {
    const removed = state.favourites.length;
    state.favourites = [];
    invalidateResults();
    saveState();
    renderFavourites();
    announceSearch(`Cleared ${removed} ${removed === 1 ? "book" : "books"} from your shelf.`);
    searchInput.focus();
  });

  form.querySelectorAll('input[type="radio"]').forEach((input) => {
    input.addEventListener("change", () => {
      invalidateResults();
      saveState();
      renderFavourites();
    });
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    requestRecommendations();
  });

  retryButton.addEventListener("click", requestRecommendations);

  restoreState();
  renderFavourites();
})();
