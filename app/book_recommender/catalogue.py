"""Curated, broad fallback catalogue for offline recommendations.

The entries intentionally use stable local keys rather than pretending to know
an Open Library work identifier. When the Search API returns the same title, its
canonical ``/works/...`` record wins during deduplication.
"""

from __future__ import annotations

import re
from typing import Any


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _book(
    title: str,
    author: str,
    year: int,
    pages: int,
    subjects: tuple[str, ...],
    popularity: float,
) -> dict[str, Any]:
    return {
        "key": f"/local/books/{_slug(title)}-{_slug(author)}",
        "title": title,
        "author_name": [author],
        "first_publish_year": year,
        # Page counts vary by edition. These values are used only to place a
        # work in a broad length band, never presented as an exact edition fact.
        "number_of_pages_median": pages,
        "subject": list(subjects),
        "language": ["eng"],
        "_curated_popularity": popularity,
    }


FALLBACK_CATALOGUE: tuple[dict[str, Any], ...] = (
    _book("The Lord of the Rings", "J. R. R. Tolkien", 1954, 1178, ("Fantasy", "Epic fiction", "Adventure", "Mythology", "Good and evil"), 1.00),
    _book("The Hobbit", "J. R. R. Tolkien", 1937, 310, ("Fantasy", "Adventure", "Dragons", "Mythology", "Quest fiction"), 0.99),
    _book("Dune", "Frank Herbert", 1965, 688, ("Science fiction", "Epic fiction", "Politics", "Ecology", "Adventure"), 1.00),
    _book("Nineteen Eighty-Four", "George Orwell", 1949, 328, ("Dystopian fiction", "Science fiction", "Politics", "Totalitarianism", "Social commentary"), 1.00),
    _book("Pride and Prejudice", "Jane Austen", 1813, 432, ("Romance", "Domestic fiction", "Social commentary", "Humorous fiction", "Classics"), 1.00),
    _book("Jane Eyre", "Charlotte Brontë", 1847, 532, ("Gothic fiction", "Romance", "Coming of age", "Identity", "Classics"), 0.96),
    _book("Frankenstein", "Mary Shelley", 1818, 280, ("Gothic fiction", "Horror", "Science fiction", "Identity", "Ethics"), 0.98),
    _book("The Handmaid's Tale", "Margaret Atwood", 1985, 320, ("Dystopian fiction", "Feminism", "Politics", "Literary fiction", "Social commentary"), 0.96),
    _book("The Name of the Wind", "Patrick Rothfuss", 2007, 662, ("Fantasy", "Epic fiction", "Magic", "Coming of age", "Adventure"), 0.94),
    _book("Mistborn: The Final Empire", "Brandon Sanderson", 2006, 541, ("Fantasy", "Magic", "Adventure", "Dystopian fiction", "Epic fiction"), 0.94),
    _book("The Way of Kings", "Brandon Sanderson", 2010, 1007, ("Fantasy", "Epic fiction", "Magic", "War", "Adventure"), 0.94),
    _book("A Game of Thrones", "George R. R. Martin", 1996, 835, ("Fantasy", "Epic fiction", "Politics", "War", "Adventure"), 0.97),
    _book("The Hunger Games", "Suzanne Collins", 2008, 374, ("Dystopian fiction", "Young adult fiction", "Adventure", "Survival", "Politics"), 0.99),
    _book("The Martian", "Andy Weir", 2011, 384, ("Science fiction", "Adventure", "Survival", "Humorous fiction", "Space exploration"), 0.96),
    _book("Project Hail Mary", "Andy Weir", 2021, 496, ("Science fiction", "Adventure", "Space exploration", "Friendship", "Humorous fiction"), 0.96),
    _book("The Three-Body Problem", "Cixin Liu", 2006, 400, ("Science fiction", "First contact", "Technology", "Philosophy", "Politics"), 0.94),
    _book("The Left Hand of Darkness", "Ursula K. Le Guin", 1969, 304, ("Science fiction", "Gender identity", "Politics", "Anthropology", "Literary fiction"), 0.91),
    _book("Station Eleven", "Emily St. John Mandel", 2014, 352, ("Dystopian fiction", "Literary fiction", "Survival", "Art", "Found family"), 0.91),
    _book("The Road", "Cormac McCarthy", 2006, 287, ("Dystopian fiction", "Literary fiction", "Survival", "Family", "Dark fiction"), 0.94),
    _book("Never Let Me Go", "Kazuo Ishiguro", 2005, 288, ("Science fiction", "Dystopian fiction", "Literary fiction", "Identity", "Memory"), 0.92),
    _book("The Night Circus", "Erin Morgenstern", 2011, 506, ("Fantasy", "Magical realism", "Romance", "Lyrical fiction", "Circus"), 0.91),
    _book("The Book Thief", "Markus Zusak", 2005, 584, ("Historical fiction", "War", "Young adult fiction", "Family", "Lyrical fiction"), 0.97),
    _book("The Shadow of the Wind", "Carlos Ruiz Zafón", 2001, 487, ("Historical fiction", "Mystery", "Gothic fiction", "Books", "Atmospheric fiction"), 0.90),
    _book("Gone Girl", "Gillian Flynn", 2012, 432, ("Thriller", "Mystery", "Crime", "Marriage", "Psychological fiction"), 0.97),
    _book("The Girl with the Dragon Tattoo", "Stieg Larsson", 2005, 672, ("Mystery", "Thriller", "Crime", "Journalism", "Psychological fiction"), 0.96),
    _book("The Thursday Murder Club", "Richard Osman", 2020, 400, ("Mystery", "Crime", "Humorous fiction", "Friendship", "Detective fiction"), 0.94),
    _book("The Seven Husbands of Evelyn Hugo", "Taylor Jenkins Reid", 2017, 400, ("Historical fiction", "Romance", "Identity", "Fame", "Character-driven fiction"), 0.97),
    _book("Normal People", "Sally Rooney", 2018, 273, ("Literary fiction", "Romance", "Coming of age", "Relationships", "Character-driven fiction"), 0.93),
    _book("The Song of Achilles", "Madeline Miller", 2011, 378, ("Historical fiction", "Mythology", "Romance", "War", "Lyrical fiction"), 0.96),
    _book("Circe", "Madeline Miller", 2018, 393, ("Fantasy", "Mythology", "Historical fiction", "Identity", "Lyrical fiction"), 0.95),
    _book("Pachinko", "Min Jin Lee", 2017, 496, ("Historical fiction", "Family saga", "Identity", "Migration", "Literary fiction"), 0.93),
    _book("A Gentleman in Moscow", "Amor Towles", 2016, 462, ("Historical fiction", "Literary fiction", "Russia", "Friendship", "Character-driven fiction"), 0.92),
    _book("The Kite Runner", "Khaled Hosseini", 2003, 371, ("Historical fiction", "Friendship", "Family", "War", "Redemption"), 0.97),
    _book("Tomorrow, and Tomorrow, and Tomorrow", "Gabrielle Zevin", 2022, 416, ("Literary fiction", "Friendship", "Technology", "Video games", "Character-driven fiction"), 0.94),
    _book("Lessons in Chemistry", "Bonnie Garmus", 2022, 400, ("Historical fiction", "Feminism", "Science", "Humorous fiction", "Family"), 0.94),
    _book("The Midnight Library", "Matt Haig", 2020, 304, ("Fantasy", "Contemporary fiction", "Mental health", "Philosophy", "Life choices"), 0.95),
    _book("Sapiens", "Yuval Noah Harari", 2011, 498, ("History", "Anthropology", "Science", "Society", "Philosophy"), 0.99),
    _book("Educated", "Tara Westover", 2018, 352, ("Memoir", "Education", "Family", "Identity", "Biography"), 0.98),
    _book("The Immortal Life of Henrietta Lacks", "Rebecca Skloot", 2010, 381, ("Biography", "Science", "Medicine", "Ethics", "History"), 0.91),
    _book("Thinking, Fast and Slow", "Daniel Kahneman", 2011, 499, ("Psychology", "Decision making", "Science", "Economics", "Human behavior"), 0.96),
    _book("Atomic Habits", "James Clear", 2018, 320, ("Self-help", "Psychology", "Habits", "Productivity", "Human behavior"), 0.99),
    _book("The Psychology of Money", "Morgan Housel", 2020, 256, ("Finance", "Psychology", "Decision making", "Business", "Human behavior"), 0.96),
    _book("Braiding Sweetgrass", "Robin Wall Kimmerer", 2013, 408, ("Nature", "Science", "Indigenous peoples", "Memoir", "Philosophy"), 0.91),
    _book("The Emperor of All Maladies", "Siddhartha Mukherjee", 2010, 571, ("Medicine", "Science", "History", "Biography", "Cancer"), 0.89),
)
