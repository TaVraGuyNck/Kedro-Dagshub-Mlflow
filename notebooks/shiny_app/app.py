from shiny import App, ui, render, reactive
from datetime import datetime
import httpx
from pathlib import Path


# values legalBasis for drop-down menu UI
mapping = {"-": " ",
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

    #title center
    ui.div(
        ui.h2("Real-time Prediction of Startup Delay - Projects Europe Horizon 2021-2027", style="text-align:center; font-weight: bold;"),
        style="margin-bottom: 30px;"
    ),

    # tab prediction start-up delay 
    ui.navset_tab(  
        ui.nav_panel("Prediction Start-up Delay",
                    ui.h4("Please Enter Following Project Details:", style="text-align: center; font-weight: bold;"),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Under which Horizon Europe Pillar falls the Project?"),
                            ui.input_select("legalBasis", " ", choices=mapping)
                        ),  
                        ui.card(
                            ui.card_header("Please provide the Country of Project's Coordinating Organization"),
                            ui.input_select("countryCoor", " ", choices=countries_dropdownmenu)
                        ), 
                        ui.card(
                            ui.card_header("Provide the Number of Participating Organizations to the Project (incl. Associated Partners)"),
                            ui.input_numeric("numberOrg"," ",value=None, min=1, step=1),
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
                                      ui.input_action_button("submit", "Submit Project Details to Generate Prediction", class_="btn btn-success"),
                                      " "
                    ), 
                    ui.layout_columns(
                        ui.output_ui("prediction_output"),style="display: flex; flex-direction"),
                        " ",      
                        ui.output_ui("validation_msg")
                    ),
        
                                                   
    
        # tab Information 
        ui.nav_panel("Information",
                    ui.h4("How the Prediction is Made:", style="text-align: center; font-weight: bold;"),
                    ui.div(
                        ui.p("A good start is half the battle! Prediction of Start-up delay for Projects under the Horizon Europe program. The aim of this project is to predict the start-up delay of Projects under Europe Horizon 2021-2027. \
                              “Start-up delay” is the delay between the date of the EC signature, and the actual start date of the project. The project invisions to mainly help program administrators identify projects “in the risk” zone for start-up delay.\
                              This identification can helpt anticipating extra and early support to these projects, such as planning timelines and setting expectations. Reducing a plausible start-up delay for projects to a minimum, will contribute to the efficiency \
                              of the Horizon program, and will enhance agility to specific project needs. The prediction is generated by a trained Machine Learning (ML) model. This model was obtained after exprimenting with various models, preprocessing techniques,\
                              different sets of features, different (hyper)parameters. The model was trained with data from all ongoing (or already ended) Horizon Europe Projects since the start in 2021."),
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
        if not input.legalBasis():
            return "Please select a legal basis for the project."
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
            pillar = input.legalBasis()
            country_coor = input.countryCoor()
            duration = input.duration()
            number_org = input.numberOrg()

            # validated input data in dictionary for api gateway
            data_to_api = {
                "totalCost": total_cost,
                "ecMaxContribution": ec_max_contribution,
                "numberOrg": number_org,
                "duration": duration,
                "pillar": pillar,
                "countryCoor": country_coor,
            }
            
            # print to help debugging -payload formed?
            print("Payload being sent to API:", data_to_api)  

            # POST to API Gateway - via httpx
            #api_url = "https://8icrl41qp8.execute-api.eu-west-3.amazonaws.com/prod/predict"
            response = httpx.post(api_url, json=data_to_api, timeout=45.0)

            # check for HTTP errors and raise exception if any and print
            # response.raise_for_status()
           # result = response.json()
            presult = {"prediction": 42}
            
            print("API response JSON:", result) 

            # receiving prediction - json format
            prediction = result.get("prediction", "No predictioon")
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
    @render.text
    def prediction_output():
        if input.submit() == 0:
            return ""
        
        result = get_prediction()
        if result is None: 
            return None

        return f"""Prediction Based On Project Details:
    ------------------------------------
   
    Europe Horizon Pillar:        {input.legalBasis()}
    Country of Coordinating 
    Organization:                 {input.countryCoor()}
    Total Cost:                   {input.totalCost()}
    EC Max Contribution:          {input.ecMaxContribution()}
    Duration (in days):           {input.duration()}
    Number of Organziations                              
    particpating in the project:  {input.numberOrg()}
    
    Predicted Startup Delay (in days): 

    {result}
    """

app = App(app_ui, server)
