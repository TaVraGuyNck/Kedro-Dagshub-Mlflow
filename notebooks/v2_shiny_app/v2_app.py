from shiny import App, ui, render, reactive
from datetime import datetime
import httpx
from pathlib import Path

# GLOBAL SCOPE — runs immediately when app.py is loaded
#api_uri = "https://8icrl41qp8.execute-api.eu-west-3.amazonaws.com/prod/predict"



# values fundingScheme for drop-down menu UI: 
fundingScheme_dropdown = {
  " ":" ",
    "HORIZON-EIC":"HORIZON-EIC",
    "HORIZON-RIA":"HORIZON-RIA",
    "HORIZON-EIC-ACC-BF":"HORIZON-EIC-ACC-BF",
    "HORIZON-CSA":"HORIZON-CSA",
    "HORIZON-IA":"HORIZON-IA",
    "HORIZON-JU-CSA":"HORIZON-JU-CSA",
    "HORIZON-COFUND":"HORIZON-COFUND",
    "HORIZON-EIC-ACC":"HORIZON-EIC-ACC",
    "HORIZON-TMA-MSCA-PF-EF":"HORIZON-TMA-MSCA-PF-EF",
    "HORIZON-JU-RIA":"HORIZON-JU-RIA",
    "HORIZON-JU-IA":"HORIZON-JU-IA",
    "EURATOM-RIA":"EURATOM-RIA",
    "HORIZON-TMA-MSCA-PF-GF":"HORIZON-TMA-MSCA-PF-GF",
    "HORIZON-TMA-MSCA-DN":"HORIZON-TMA-MSCA-DN",
    "HORIZON-TMA-MSCA-SE":"HORIZON-TMA-MSCA-SE",
    "HORIZON-TMA-MSCA-Cofund-P":"HORIZON-TMA-MSCA-Cofund-P",
    "MSCA-PF":"MSCA-PF",
    "HORIZON-TMA-MSCA-Cofund-D":"HORIZON-TMA-MSCA-Cofund-D",
    "HORIZON-TMA-MSCA-DN-JD":"HORIZON-TMA-MSCA-DN-JD",
    "HORIZON-TMA-MSCA-DN-ID":"HORIZON-TMA-MSCA-DN-ID",
    "EURATOM-IA":"EURATOM-IA",
    "EURATOM-CSA":"EURATOM-CSA",
    "RIA":"RIA",
    "HORIZON-AG":"HORIZON-AG",
    "CSA":"CSA",
    "HORIZON-AG-UN":"HORIZON-AG-UN",
    "HORIZON-ERC-POC":"HORIZON-ERC-POC",
    "HORIZON-ERC":"HORIZON-ERC",
    "EIC":"EIC",
    "HORIZON-EIT-KIC":"HORIZON-EIT-KIC",
    "HORIZON-PCP":"HORIZON-PCP",
    "HORIZON-ERC-SYG":"HORIZON-ERC-SYG",
    "EURATOM-COFUND":"EURATOM-COFUND",
    "ERC":"ERC",
    "HORIZON-AG-LS":"HORIZON-AG-LS",
    "ERC-POC":"ERC-POC"
}


# values pillar for drop-down menu UI
mapping = {
    "-": " ",
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
    " ": " ",
    "European Union Member States":{
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
app_ui = ui.page_fillable( 
    ui.include_css(Path(__file__).parent / "style.css"),

    ui.div(
        ui.h2("Real-time Prediction of Startup Delay - Projects Europe Horizon 2021-2027", style="text-align:center; font-weight: bold;"),
        style="margin-bottom: 30px;"
    ),

    ui.navset_tab(
        ui.nav_panel(
            "Prediction Start-up Delay",
            ui.h4("Please Enter Following Project Details:", style="text-align: center; font-weight: bold;"),

            ui.layout_columns(
                ui.card(
                    ui.card_header("Under which Horizon Europe Pillar falls the Project?"),
                    ui.input_select("pillar", " ", choices=mapping)
                ),  
                ui.card(
                    ui.card_header("Please provide the Country of Project's Coordinating Organization"),
                    ui.input_select("countryCoor", " ", choices=countries_dropdownmenu)
                ), 
                ui.card(
                    ui.card_header("Provide the Number of Participating Organizations to the Project (incl. Associated Partners)"),
                    ui.input_numeric("numberOrg", " ", value=None, min=1, step=1),
                ),
            ),

            ui.layout_columns(
                ui.card(
                    ui.card_header("Please provide foreseen Total Cost of the Project"),
                    ui.input_numeric("totalCost", " ", value=None, min=0.000001, step=1),
                ),
                ui.card(
                    ui.card_header("Please provide foreseen Maximum EU Contribution to the Project"),
                    ui.input_numeric("ecMaxContribution", " ", value=None, min=0.000001, step=1)
                ),
                ui.card(
                    ui.card_header("Please provide foreseen Duration of the Project (in days)"),
                    ui.input_numeric("duration", " ", value=None, min=0, step=1),
                )
            ),
            ui.layout_columns(" ",
                              ui.card(
                                  ui.card_header("Funding Scheme of the Project"),
                                  ui.input_select("fundingScheme", " ", choices=fundingScheme_dropdown)
                              ),
                              "  "
            ),

            ui.layout_columns(" ",
                ui.input_action_button("submit", "Submit Project Details to Generate Prediction", class_="btn btn-success"),
                "  "),
            ui.layout_columns(" "
            ),
          
            ui.layout_columns(" ",
                              ui.output_ui("validation_msg"),
                              " "
            ),
    
            ui.tags.div(
                {"class": "background"},
                ui.tags.div(
                    ui.output_ui("prediction_output"),class_="text-block", style="top: 50px; right 60%; width : 70%;text_align:left")
        ),
    ),

        ui.nav_panel("Information",
            ui.h4("How the Prediction is Made:", style="text-align: center; font-weight: bold;"),
            ui.div(
                ui.p(
                    "A good start is half the battle! Prediction of Start-up delay for Projects under the Horizon Europe program. "
                    "The aim of this project is to predict the start-up delay of Projects under Europe Horizon 2021-2027. "
                    "“Start-up delay” is the delay between the date of the EC signature, and the actual start date of the project. "
                    "The project envisions to mainly help program administrators identify projects “in the risk” zone for start-up delay. "
                    "This identification can help anticipating extra and early support to these projects, such as planning timelines and setting expectations. "
                    "Reducing a plausible start-up delay for projects to a minimum will contribute to the efficiency of the Horizon program, and will enhance agility "
                    "to specific project needs. The prediction is generated by a trained Machine Learning (ML) model. This model was obtained after experimenting with "
                    "various models, preprocessing techniques, different sets of features, and hyperparameters. The model was trained with data from all ongoing (or already ended) "
                    "Horizon Europe Projects since the start in 2021."
                ),
                style="text-align: center; margin-top: 20px;"
            )
        )
    )
    )

# defining steps for the server
def server(input, output, session):

    @reactive.calc
    def validate_inputs():
        if input.totalCost() is None:
            return "Please enter Total Cost"
        if input.totalCost() <= 0:
            return "Total cost must be greater than 0."
        if input.ecMaxContribution() is None:
            return "Please enter Maximum EU Contribution"
        if input.ecMaxContribution() <= 0:
            return "Maximum EU contribution must be greater than 0."
        if input.totalCost() < input.ecMaxContribution():
            return "Maximum EU contribution cannot be greater than total cost of the project."
        if not input.countryCoor():
            return "Please select a country for Coordinating Organization."
        if not input.pillar():
            return "Please select the correct pillar to which the project belongs."
        if not input.fundingScheme(): 
            return "Please selcte the applicable funding scheme for the project."
        if input.duration() is None:
            return "Please enter the duration of the project in days."
        if not isinstance(input.duration(), int):
            return "Duration cannot be a decimal value."
        if input.duration() <= 0:
            return "Project duration cannot be 0."
        if input.numberOrg() is None:
            return "Please enter the number of participating organizations."
        if not isinstance(input.numberOrg(), int):
            return "Number of participating organizations cannot be a decimal value."
        if input.numberOrg() <= 0:
            return "Number of participating organizations must be greater than 0."

        return ""

    
    
    @output
    @render.ui
    def validation_msg():
        msg = validate_inputs()
        if input.submit() == 0 or not msg:
            return ""
        return ui.tags.div({"style": "color:red;"}, f" {msg}")
    

    # processing input after user clicks "submit"
    @reactive.event(input.submit)
    def get_prediction():
        msg = validate_inputs()
        if msg:
            return None

        try: 
            #input saved as variables
            total_cost = input.totalCost()
            ec_max_contribution = input.ecMaxContribution()
            pillar = input.pillar()
            country_coor = input.countryCoor()
            duration = input.duration()
            number_org = input.numberOrg()
            fundingScheme = input.fundingScheme()

            # validated input data in dictionary for api gateway
            data_to_api = {
                "totalCost": total_cost,
                "ecMaxContribution": ec_max_contribution,
                "numberOrg": number_org,
                "duration": duration,
                "pillar": pillar,
                "countryCoor": country_coor,
                "fundingScheme": fundingScheme
            }
            
            # print to help debugging -payload formed?
            print("Payload being sent to API:", data_to_api)  

            # POST to API Gateway - via httpx
            response = httpx.post(api_uri, json=data_to_api, timeout=45.0)

            # check for HTTP errors and raise exception if any and print
            response.raise_for_status()
            result = response.json()
          
            
            print("API response JSON:", result) 

            # receiving prediction - json format
            prediction = result.get("prediction", "No prediction")
            if prediction is None:
                return "No prediction was returned. Response: {}".format(result)
            else:
                return f"Predicted Startup Delay (in days):\n\n{prediction}"

        # handle HTTP errors and other exceptions
        except httpx.HTTPStatusError as status_error:
            return f"HTTP error occurred: {status_error.response.status_code} - {status_error.response.text}"

        except httpx.RequestError as request_error:
            return f"Request error occurred: {str(request_error)}"
        
        except Exception as exc:
            return f"Unexpected error: {str(exc)}"


    # calling get_predition function to display prediction on UI
    @output
    @render.ui
    def prediction_output():
        if input.submit() == 0:
            return ""
        
        result = get_prediction()
        if result is None: 
            return None

        return ui.tags.div(
            {"style": "white-space: pre-wrap;"},
        f"""Prediction based on the Project details provided: 
-------------------------------------------------------------------------------------

RESULT: {result}
__________________________________________________________

Europe Horizon Pillar--------------------------------------------- {input.pillar()}
Country of Coordinating Organization:----------------------------- {input.countryCoor()}
Total Cost:-------------------------------------------------------- {input.totalCost()}
EC Max Contribution:------------------------------------------------{input.ecMaxContribution()}
Duration (in days):------------------------------------------------{input.duration()}
Number of Participating Organizations:-----------------------------{input.numberOrg()}



""")
       
app = App(app_ui, server)