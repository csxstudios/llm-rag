import re
import bs4, requests
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from dotenv import load_dotenv
import helpers

load_dotenv()

SCRAPE_PATH="data/scrape/taggs_website"
urls = [
    {
        "url": "https://taggs.hhs.gov/",
        "page": "Home"
    },
    {
        "url": "https://taggs.hhs.gov/About",
        "page": "About"
    },
    {
        "url": "https://taggs.hhs.gov/About/Data_Dictionary",
        "page": "Data Dictionary"
    },
    {
        "url": "https://taggs.hhs.gov/About/FAQs",
        "page": "FAQs"
    },
    {
        "url": "https://taggs.hhs.gov/Overview/Search",
        "page": "Search Overview"
    },
    {
        "url": "https://taggs.hhs.gov/Overview/Report",
        "page": "Reports Overview"
    },
    {
        "url": "https://taggs.hhs.gov/Coronavirus/Overview",
        "page": "HHS COVID-19 Funding Overview"
    },
    {
        "url": "https://taggs.hhs.gov/TotalAssist",
        "page": "Total Assistance"
    },
    {
        "url": "https://taggs.hhs.gov/OtherFinancialAssist",
        "page": "Other Financial Assistance"
    },
    {
        "url": "https://taggs.hhs.gov/DataQuality/SubmissionStatus",
        "page": "Data Submission Status"
    },
    {
        "url": "https://taggs.hhs.gov/ReportsGrants/GrantsByOPDIV",
        "page": "Grants By OPDIV"
    },
    {
        "url": "https://taggs.hhs.gov/ReportsGrants/GrantsByRecipClass",
        "page": "Grants By Recipient Class"
    },
    {
        "url": "https://taggs.hhs.gov/ReportsGrants/GrantsbyActivityType",
        "page": "Grants By Activity Type"
    },
    {
        "url": "https://taggs.hhs.gov/ReportsLocation/GrantsbyLocationIndex",
        "page": "Grants By Location for both US and World"
    },
    {
        "url": "https://taggs.hhs.gov/ReportsLocation/GrantsByLocation_MetroNonmetro",
        "page": "Grants By Location for US by Metropolitan and Non-metropolitan"
    },
    {
        "url": "https://taggs.hhs.gov/ReportsCFDA",
        "page": "Assistance Listings Summary Report"
    },
    {
        "url": "https://taggs.hhs.gov/ReportsRecipientSummary",
        "page": "Recipient Summary Report"
    },
    {
        "url": "https://taggs.hhs.gov/ReportsAwardSummary",
        "page": "Award Summary Report"
    },
    {
        "url": "https://taggs.hhs.gov/Coronavirus",
        "page": "COVID-19 Awards"
    },
    {
        "url": "https://taggs.hhs.gov/Coronavirus/Providers",
        "page": "Provider Relief Fund"
    },
    {
        "url": "https://taggs.hhs.gov/Coronavirus/RuralHealthClinics",
        "page": "Rural Health Clinic (RHC) Testing & Mitigation"
    },
    {
        "url": "https://taggs.hhs.gov/Coronavirus/Uninsured",
        "page": "Testing, Treatment, and Vaccine Administration for the Uninsured"
    },
    {
        "url": "https://taggs.hhs.gov/Coronavirus/CoverageAssistanceFund",
        "page": "Coverage Assistance Fund"
    }
]

def scrape_url(url, name):
    bs_strainer = bs4.SoupStrainer(role=["main"])
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": bs_strainer},
    )
    docs = loader.load()
    # print(docs)

    file_name = name.replace(' ', '-').lower()
    
    full_file_path = f"{SCRAPE_PATH}/taggs_website_{file_name}.txt"
    # full_file_path = f"data/taggs_{name}.txt"

    for page in docs:
        page_cleanup = re.sub(r'[\r\n\t]|(\s\s\s\s\s)', ' ', page.page_content.strip())
        page_cleanup = re.sub(r'(\s\s\s\s\s\s\s\s)', '', page_cleanup)
        page_cleanup = re.sub(r'([^\x00-\x7F])', '', page_cleanup)
        # content =f"Text from the TAGGS website's {name} page:\n" + page_cleanup
        # content+=f"\n\n"
        # print(content)
        with open(full_file_path, "a") as text_file:
            text_file.write(page_cleanup)

    return full_file_path

docs = []

for url in urls:
    print(url, url["url"])
    full_file_path = scrape_url(url["url"], url["page"])
    doc = helpers.split_text_to_doc(full_file_path, url)
    docs += doc
    # print(docs)

print(len(docs))

with open(f'{SCRAPE_PATH}/docs_list.txt', 'w') as f:
    for line in docs:
        f.write(f"{line}\n")

helpers.docs_to_chroma(docs,True)