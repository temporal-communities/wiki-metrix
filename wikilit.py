import argparse
import pywikibot
import requests
import polars as pl
from tqdm import tqdm

version = "0.0.4"


def get_page_stats(page: pywikibot.Page):
    """
    Get page stats for a given page.
    """
    page_content = page.get(force=True)
    length_in_bytes = len(page_content.encode("utf-8"))
    page_revisions = list(page.revisions(reverse=True))

    data = {
        "title": page.title(),
        "url": page.full_url(),
        "length": length_in_bytes,
        "n_contributors": len(page.contributors()),
        "n_revisions": len(page_revisions),
        "n_extlinks": len(list(page.extlinks())),
        "n_langlinks": len(page.langlinks()),
        "n_links": len(list(page.linkedPages())),
        "n_linkshere": len(
            list(page.linkedPages(namespaces=[0], follow_redirects=False))
        ),  # Article namespace only (0)
        "n_categories": len(list(page.categories())),
        "pageviews_60d": get_pageviews(page),
        "first_revision": page_revisions[0].timestamp,
    }

    MW_API_LIMIT = 500
    # Give warning if any value is at the limit
    for key, value in data.items():
        if value == MW_API_LIMIT:
            print(f"Warning: {key} at limit {MW_API_LIMIT}.")

    return data


# Uses pageview api to count page views in time range for each page respectively
def get_pageviews(page: pywikibot.Page):
    lang = page.site.code
    site = page.site.family.name

    # Construct the API URL for pageviews
    url = f"https://{lang}.{site}.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "pageviews",
        "titles": page.title(),
        "pvipdays": "60",  # Number of days for pageview count
    }

    # Make the API request
    response = requests.get(url, params=params)
    data = response.json()

    page_data = data["query"]["pages"][str(page.pageid)]

    # Filter out None values and sum
    pageviews_sum = sum(filter(None, page_data["pageviews"].values()))

    return pageviews_sum


def collect_page_stats(cases: dict[str, pywikibot.Page]):
    """
    Collect page stats for a given list of pages.
    """
    dicts = []
    for case_label, page in (pbar := tqdm(cases.items(), desc="Getting page stats")):
        pbar.set_postfix_str(case_label)
        stats = get_page_stats(page)
        dicts.append(dict({"selection_case": case_label}, **stats))

    # print stats
    df = pl.from_dicts(dicts)
    return df


# Convert arguments to named parameters
def wikilit(
    selection_method: str,
    selection: str,
    lang="de",
    site="wikipedia",
    article_title_column="article",
):
    input_data = None

    # Error handling
    allowed_selection_types = ["category", "langlinks", "file"]
    if selection_method not in allowed_selection_types:
        raise ValueError(
            f"selection_method must be one of {allowed_selection_types}, but is {selection_method}."
        )
    if selection is None or selection == "":
        raise ValueError("selection must be a non-empty string.")

    mw_site = pywikibot.Site(lang, site)

    # Main logic
    cases = {}
    if selection_method == "category":
        category_name = selection

        # Given the (German) name of a category, extract statistics
        # for all articles belonging to that category.
        category = pywikibot.Category(mw_site, category_name)
        pages = category.articles(recurse=True)

        cases = {page.title(): page for page in pages}

    elif selection_method == "langlinks":
        # Given the (German) name of an article, extract statistics
        # for all available language editions.
        page_title = selection
        page = pywikibot.Page(mw_site, page_title)

        langlinks = page.langlinks()
        # Sort keys
        langlinks = sorted(langlinks, key=lambda k: k.site.code)

        # Create dict with langlink[lang] as key and page as value
        cases = {
            f"{langlink.site.code}:{langlink.title}": pywikibot.Page(
                langlink.site, langlink.title
            )
            for langlink in langlinks
        }

        # Add original page
        cases[f"{page.site.code}:{page.title()}"] = page

    elif selection_method == "file":
        file = selection
        input_data = pl.read_csv(file, separator="\t")

        # Construct cases
        cases = {
            row[article_title_column]: pywikibot.Page(
                mw_site, row[article_title_column]
            )
            for row in input_data.to_dicts()
        }

    # Error if no cases
    if len(cases) == 0:
        raise ValueError("No cases found.")

    # Get page stats for all cases
    stats = collect_page_stats(cases)
    meta_columns = ["selection_method", "selection", "selection_case"]
    stats = (
        # Add selection method and selection value as columns
        stats.with_columns(
            selection_method=pl.lit(selection_method),
            selection=pl.lit(selection),
        )
        # Move meta columns to end
        .select(
            [col for col in stats.columns if col not in meta_columns] + meta_columns
        )
    )

    # If selection_method is file, add stats to original data
    if selection_method == "file":
        assert input_data is not None
        # Join data
        stats = input_data.join(
            stats, left_on=article_title_column, right_on="title", how="left"
        )

    return stats


# main method - program control flow starts here
if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract stats from Wikipedia",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # selection method
    select_command = parser.add_mutually_exclusive_group(required=True)
    select_command.add_argument(
        "-l",
        "--langlinks",
        type=str,
        metavar="ART",
        help="Use langlinks selection method on article",
    )
    select_command.add_argument(
        "-c",
        "--category",
        type=str,
        metavar="CAT",
        help="Use category selection method on category",
    )
    select_command.add_argument(
        "-f",
        "--file",
        type=argparse.FileType("r", encoding="utf-8"),
        metavar="FILE",
        help="Use file selection method on file",
    )
    parser.add_argument(
        "-L",
        "--lang",
        type=str,
        metavar="LANG",
        help="language edition to be used",
        default="en",
    )
    parser.add_argument(
        "-S",
        "--site",
        type=str,
        metavar="SITE",
        help="site to be used",
        default="wikipedia",
    )
    parser.add_argument(
        "-v", "--version", action="version", version="%(prog)s " + version
    )
    args = parser.parse_args()

    global_arguments = [args.lang, args.site]
    # Call appropriate function
    if args.langlinks:
        wikilit(
            selection_method="langlinks", selection=args.langlinks, *global_arguments
        )
    elif args.category:
        wikilit(selection_method="category", selection=args.category, *global_arguments)
    elif args.file:
        wikilit(selection_method="file", selection=args.file, *global_arguments)
    else:
        parser.print_help()
