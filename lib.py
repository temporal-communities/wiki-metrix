import argparse
import datetime
import urllib.parse

import polars as pl
import pywikibot
import requests
from tqdm import tqdm

version = "0.0.4"
user_agent = f"wiki-metrix (https://github.com/temporal-communities/wiki-metrix) requests/{requests.__version__}"


def get_page_stats(page: pywikibot.Page):
    """
    Get page stats for a given page.
    """

    # Handle redirects
    page = handle_redirect(page)

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
        "pageviews_365d": get_pageviews(page),
        "first_revision": page_revisions[0].timestamp,
    }

    MW_API_LIMIT = 500
    # Give warning if any value is at the limit
    for key, value in data.items():
        if value == MW_API_LIMIT:
            print(f"Warning: {key} at limit {MW_API_LIMIT}.")

    return data


# Use Wikimedia Pageviews REST API to get pageviews
def get_pageviews(page: pywikibot.Page):
    lang = page.site.code
    site = page.site.family.name

    # Wikimedia REST API
    # https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews
    # https://wikimedia.org/api/rest_v1/
    end_date = datetime.date.today() - datetime.timedelta(days=2)  # Two days ago
    start_date = end_date - datetime.timedelta(days=365)  # Two days minus one year ago

    agent_type = "user"  # user, bot, spider, all-agents
    title_uri = urllib.parse.quote(
        page.title(underscore=True, with_section=False), safe=""
    )  # URI-encoded title, no safe characters
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.{site}/all-access/{agent_type}/{title_uri}/monthly/{start_date.strftime('%Y%m%d')}/{end_date.strftime('%Y%m%d')}"

    response = requests.get(url, headers={"User-Agent": user_agent})

    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} {response.reason}")

    data = response.json()
    pageviews_sum = sum(filter(None, [item["views"] for item in data["items"]]))

    return pageviews_sum


def create_meta_dict(label: str) -> dict[str, str]:
    """
    Create a meta dictionary with label and timestamp.
    """
    utc_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return {"selection_case": label, "timestamp": utc_timestamp}


def collect_page_stats(cases: dict[str, pywikibot.Page]):
    """
    Collect page stats for a given list of pages.
    """
    dicts = []
    null_cases = {k: v for k, v in cases.items() if v is None}
    non_null_cases = {k: v for k, v in cases.items() if v is not None}

    # Add null cases
    for case_label in null_cases.keys():
        dicts.append(create_meta_dict(case_label))

    # Fetch stats for non-null cases
    for case_label, page in (
        pbar := tqdm(non_null_cases.items(), desc="Getting page stats")
    ):
        pbar.set_postfix_str(page.title())
        stats = get_page_stats(page)
        dicts.append(dict(create_meta_dict(case_label), **stats))

    df = pl.from_dicts(dicts)
    return df


def handle_redirect(page: pywikibot.Page):
    """
    Check if page is a redirect and return target page.
    """
    if page.isRedirectPage():
        redirect_title = page.title(underscore=True)
        page = page.getRedirectTarget()
        print(
            f"Warning: Page {redirect_title} is a redirect to {page.title(underscore=True)}."
        )
    return page


def wikidata_qids_to_titles(qids: pl.Series, site: pywikibot.BaseSite) -> pl.Series:
    site_id = site.code + "wiki"  # e.g. "enwiki"

    titles = []
    no_sitelink = []
    for qid in (pbar := tqdm(qids, desc="Getting article titles from Wikidata")):
        pbar.set_postfix_str(qid)

        url = f"https://wikidata.org/w/rest.php/wikibase/v0/entities/items/{qid}/sitelinks/{site_id}"
        res = requests.get(url, headers={"User-Agent": user_agent})
        json = res.json()

        if res.status_code != 200:
            if json["code"] == "sitelink-not-defined":
                no_sitelink.append(qid)
                titles.append(None)
                continue

            raise Exception(f"Error: {res.status_code} {res.reason}")

        # Example response
        # {
        #     "badges": [],
        #     "title": "Douglas Adams",
        #     "url": "https://de.wikipedia.org/wiki/Douglas_Adams"
        # }

        titles.append(json["title"])

    if len(no_sitelink) > 0:
        print(
            f"Warning: No sitelink found on {site.code}.{site.family.name} for {', '.join(no_sitelink)}."
        )

    return pl.Series(titles)


# Convert arguments to named parameters
def wikimetrix(
    selection_method: str,
    selection: str,
    lang="de",
    site="wikipedia",
    input_column="article",
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

        # Given the name of a category, extract statistics
        # for all articles belonging to that category.
        category = pywikibot.Category(mw_site, category_name)
        pages = category.articles(recurse=False)

        cases = {page.title(): page for page in pages}

    elif selection_method == "langlinks":
        # Given the name of an article, extract statistics
        # for all available language editions.
        page_title = selection
        page = pywikibot.Page(mw_site, page_title)

        # Handle redirects
        # Must be done here to properly get langlinks
        page = handle_redirect(page)

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

        # Check if input_column contains Wikidata Q-IDs as URL
        if all(
            input_data[input_column].str.contains(
                r"^http://www.wikidata.org/entity/Q\d+$"
            )
        ):
            # Remove URL prefix
            input_data = input_data.with_columns(
                pl.col(input_column)
                .str.replace(r"^http://www.wikidata.org/entity/", "")
                .keep_name()
            )

        # Check if input_column contains Wikidata Q-IDs
        if all(input_data[input_column].str.contains(r"^Q\d+$")):
            util_title_column = "util_article_title"
            input_data = input_data.with_columns(
                pl.col(input_column)
                .map_batches(
                    lambda qids: wikidata_qids_to_titles(qids, mw_site),
                    # pl.String | None,
                )
                .alias(util_title_column)
            )

            # Construct cases
            cases = {}
            for row in input_data.to_dicts():
                if row[util_title_column] is not None:
                    cases[row[input_column]] = pywikibot.Page(
                        mw_site, row[util_title_column]
                    )
                else:
                    cases[row[input_column]] = None

            # Remove utility column
            input_data.drop_in_place(util_title_column)

        else:
            # Construct cases
            cases = {
                row[input_column]: pywikibot.Page(mw_site, row[input_column])
                for row in input_data.to_dicts()
            }

    # Error if no cases
    if len(cases) == 0:
        raise ValueError("No cases found.")

    # Get page stats for all cases
    stats = collect_page_stats(cases)

    meta_columns = [
        "label",
        "timestamp",
        "selection_method",
        "selection",
        "selection_case",
    ]
    stats = (
        # Add selection method and selection value as columns
        stats.with_columns(
            selection_method=pl.lit(selection_method),
            selection=pl.lit(selection),
            # Create a label column, e.g. "Q123/Title" if selection_case is a Q-ID
            label=pl.when(pl.col("selection_case").str.contains(r"^Q\d+$"))
            .then(pl.col("selection_case") + pl.lit("/") + pl.col("title"))
            .otherwise(pl.col("selection_case")),
        )
    )

    # If selection_method is file, add stats to original data
    if selection_method == "file":
        assert input_data is not None
        stats = input_data.join(
            # Join data
            stats,
            left_on=input_column,
            right_on="selection_case",
            how="left",
        ).with_columns(
            # Re-add selection_case column
            selection_case=pl.col(input_column)
        )

    # Move meta columns to end
    stats = stats.select(
        [col for col in stats.columns if col not in meta_columns] + meta_columns
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
        wikimetrix(
            selection_method="langlinks", selection=args.langlinks, *global_arguments
        )
    elif args.category:
        wikimetrix(
            selection_method="category", selection=args.category, *global_arguments
        )
    elif args.file:
        wikimetrix(selection_method="file", selection=args.file, *global_arguments)
    else:
        parser.print_help()
