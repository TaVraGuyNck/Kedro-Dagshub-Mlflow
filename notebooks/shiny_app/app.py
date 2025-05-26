from shiny import App, ui, render, reactive
import requests
from datetime import datetime
import httpx

# set values legalBasis for drop-down menu UI
mapping = {"": "",
    "Pillar 1 - European Research Council (ERC)": "HORIZON.1.1 - Pillar 1 - European Research Council (ERC)",
    "Pillar 1 - Marie Sklodowska-Curie Actions (MSCA)": "HORIZON.1.2 - Pillar 1 - Marie Sklodowska-Curie Actions (MSCA)",
    "Pillar 1 - Research infrastructures": "HORIZON.1.3 - Pillar 1 - Research infrastructures",
    "Pillar 2 - Health": "HORIZON.2.1 - Pillar 2 - Health",
    "Pillar 2 - Culture, creativity and inclusive society": "HORIZON.2.2 - Pillar 2 - Culture, creativity and inclusive society",
    "Pillar 2 - Civil Security for Society": "HORIZON.2.3 - Pillar 2 - Civil Security for Society",
    "Pillar 2 - Digital, Industry and Space": "HORIZON.2.4 - Pillar 2 - Digital, Industry and Space",
    "Pillar 2 - Climate, Energy and Mobility": "HORIZON.2.5 - Pillar 2 - Climate, Energy and Mobility",
    "Pillar 2 - Food, Bioeconomy Natural Resources, Agriculture and Environment": "HORIZON.2.6 - Pillar 2 - Food, Bioeconomy Natural Resources, Agriculture and Environment",
    "Pillar 3 - The European Innovation Council (EIC)": "HORIZON.3.1 - Pillar 3 - The European Innovation Council (EIC)",
    "Pillar 3 - European innovation ecosystems": "HORIZON.3.2 - Pillar 3 - European innovation ecosystems",
    "Pillar 3 - Cross-cutting call topics": "HORIZON.3.3 - Pillar 3 - Cross-cutting call topics",
    "Improve and support nuclear safety...": "EURATOM.1.1 - Improve and support nuclear safety...",
    "Maintain and further develop expertise...": "EURATOM.1.2 - Maintain and further develop expertise...",
    "Foster the development of fusion energy...": "EURATOM.1.3 - Foster the development of fusion energy...",
    "EURATOM2027": "EURATOM2027"
}

# set values for countryCoor for drop-down menu UI - 2 sections (EU memberstates, Associated countries)
countries_dropdownmenu = {
    "": "",
    "European Union Member States": {
        "AT": "Austria",
        "BE": "Belgium",
        "BG": "Bulgaria",
        "HR": "Croatia",
        "CY": "Cyprus",
        "CZ": "Czech Republic",
        "DK": "Denmark",
        "EE": "Estonia",
        "FI": "Finland",
        "FR": "France",
        "DE": "Germany",
        "GR": "Greece",
        "HU": "Hungary",
        "IE": "Ireland",
        "IT": "Italy",
        "LV": "Latvia",
        "LT": "Lithuania",
        "LU": "Luxembourg",
        "MT": "Malta",
        "NL": "Netherlands",
        "PL": "Poland",
        "PT": "Portugal",
        "RO": "Romania",
        "SK": "Slovakia",
        "SI": "Slovenia",
        "ES": "Spain",
        "SE": "Sweden"
    },
    "Associated Countries": {
        "AL": "Albania",
        "AM": "Armenia",
        "BA": "Bosnia and Herzegovina",
        "FO": "Faroe Islands",
        "GE": "Georgia",
        "IS": "Iceland",
        "IL": "Israel",
        "XK": "Kosovo",
        "MD": "Moldova",
        "ME": "Montenegro",
        "NZ": "New Zealand",
        "MK": "North Macedonia",
        "NO": "Norway",
        "RS": "Serbia",
        "TN": "Tunisia",
        "TR": "Türkiye",
        "UA": "Ukraine",
        "GB": "United Kingdom"
    }
}

# UI definition 
app_ui = ui.page_fluid(

    # header picture
    ui.div(
        ui.img(src="horizon_banner.png", width="100%", height="auto"),
        style="margin-bottom: 20px;"
    ),

    #title center
    ui.div(
        ui.h2("Prediction of Startup Delay - Projects Europe Horizon 2021-2027", style="text-align:center; font-weight: bold;"),
        style="margin-bottom: 30px;"
    ),

   # two-Column Layout
    ui.layout_columns(

        # Left column: Paragraph/text content
        ui.column(6,
            ui.div(
                ui.p("""
                    text text text  
                    text text text text  
                    text text text text text  
                    text text text text text text text text text text text text  
                    text text vvvvtext text text text text text text text text text  
                    text text text text text text text text text text text text  
                    text text text text text text text text text text text text  
                    text text text text text text text text text text text text"""),
                style="padding-right: 20px;"
            ),

        # middle part
            ui.div(
                ui.h5("Prediction Output Startdelay"),
                ui.output_text_verbatim("prediction_output", placeholder=True),
                style="margin-top: 10px;"
            )
        ),

        # right column - data input fields, title, submit button
        ui.column(6,
            ui.div(
                ui.h4("Please Enter Following Details of the Project:", style="text-align: center; font-weight: bold;"),
                    ui.layout_columns(
                        ui.input_select("legalBasis", "Europe Horizon 2021-2027 Pillar", choices=mapping),
                        ui.input_select("countryCoor", "Country of Project's Coordinating Organization", choices=countries_dropdownmenu),
                        ui.input_numeric("totalCost", "Foreseen Total Cost of Project", value=None, min=0),
                        ui.input_numeric("duration", "Foreseen Duration of Project (in days)", value=None, min=0),
                        ui.input_numeric("ecMaxContribution", "Foreseen Maximum EU Contribution for Project:", value=None, min=0),
                        ui.input_numeric("numberOrg", "Number of Participating Organizations in the Project:", value=None, min=0),
                        col_widths=[6, 6]  # Split form into 2 equal halves
                    ),
                    ui.input_action_button("submit", "Submit"),
                    style="display: flex; flex-direction: column; gap: 15px;"
                )   
            )
        )
)

# defining steps for the server
def server(input, output, session):

    # processing input after user clicks "submit"
    @reactive.event(input.submit)
    def get_prediction():

        try:
            # Validate and extract inputs
            total_cost = input.totalCost()
            ec_max_contribution = input.ecMaxContribution()
            pillar = input.legalBasis()
            country_coor = input.countryCoor()
            duration = input.duration()

            # Validate numberOrg
            number_org = input.numberOrg()
            if number_org is None or number_org < 1:
                return "Please provide the number of participating organizations for your project, as reflected in your project proposal."
    
            # all input and calculated data in dictionary for api gateway
            payload = {
                "totalCost": total_cost,
                "ecMaxContribution": ec_max_contribution,
                "numberOrg": number_org,
                "duration": duration,
                "pillar": pillar,
                "countryCoor": country_coor,
            }

            print("Payload being sent to API:", payload)  # Debug print

            # sending to API Gateway - via httpx
            api_url = "https://8icrl41qp8.execute-api.eu-west-3.amazonaws.com/prod/predict"
            response = httpx.post(api_url, json=payload, timeout=40.0)

            # Check for HTTP errors and raise exception if any
            response.raise_for_status()

            result = response.json()
            print("API response JSON:", result)  # Debug print

            # receiving prediction - json
            prediction = result.get("prediction")
            if prediction is None:
                return f"API returned no prediction. Full response: {result}"

            return f"Predicted startup delay (days): {prediction}"
    
        except httpx.HTTPStatusError as http_err:
            # Handle HTTP errors separately to show status code and message
            return f"HTTP error occurred: {http_err.response.status_code} - {http_err.response.text}"

        except Exception as e:
            # General catch-all for other exceptions
            return f"Unexpected error: {str(e)}"

    # calling get_predition function to display prediction on UI
    @output
    @render.text
    def prediction_output():
        result = get_prediction()

        # If the result is an error message, show it directly
        if isinstance(result, str) and not result.startswith("Predicted"):
            return result

        # Compose a summary with input values + prediction
        return f"""
    Prediction Summary:
    -------------------
    Europe Horizon Pillar: {input.legalBasis()}
    Country Coordinator:   {input.countryCoor()}
    Total Cost:            {input.totalCost()}
    EU Max Contribution:   {input.ecMaxContribution()}
    Duration (days):       {input.duration()}
    Number of Orgs:        {input.numberOrg()}

    {result}
    """
    

app = App(app_ui, server)