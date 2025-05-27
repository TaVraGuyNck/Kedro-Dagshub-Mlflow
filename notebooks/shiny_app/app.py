from shiny import App, ui, render, reactive
import requests
from datetime import datetime
import httpx
from pathlib import Path

# set values legalBasis for drop-down menu UI
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

# defining custom input validation functions
def input_field_filled(value):
    if value is None or value == "":
        return "Please ensure to fill all required details on the project, in order to generate a prediction."
    return None

def input_field_not_integer(value):
    if value != int(value)
        return "Number or Organizations participating to the project and Duration in days cannot be set as decimal numbers."
    return None

def input_field_not_zero(value):
    if value == 0:
        return "Total Cost, Maximum EU Contribution, Duration of the project in days and Number of Organizations paticipating to the project cannot be negative or 0."
    return None

# UI definition 
app_ui = ui.page_fillable( 
    ui.include_css(Path(__file__).parent / "styles.css"),                     

    #title center
    ui.div(
        ui.h2("Prediction of Startup Delay - Projects Europe Horizon 2021-2027", style="text-align:center; font-weight: bold;"),
        style="margin-bottom: 30px;"
    ),

    # tab prediction start-up delay 
    ui.navset_tab(  
        ui.nav_panel("Prediction Start-up Delay",
                    ui.h4("Please Enter Following Project Details:", style="text-align: center; font-weight: bold;"),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Under which Horizon Europe Pillar falls the Project?"),
                            ui.input_select("legalBasis", "legalBasis", " ", choices=mapping)
                        ),  
                        ui.card(
                            ui.card_header("Please provide the Country of Project's Coordinating Organization"),
                            ui.input_select("countryCoor","countryCoor", " ", choices=countries_dropdownmenu)
                        ), 
                        ui.card(
                            ui.card_header("Provide the Number of Participating Organizations to the Project (incl. Associated Partners)"),
                            ui.input_numeric("numberOrg", "numberOrg"," ",value=None, min=1, step=1),
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

                    ui.layout_column_wrap(
                        ui.output_text_verbatim("prediction_output", 
                                                style="text-align: center; margin-top:20px; color:darkgreen"
                                                )
                    ),
                    ui.layout_column_wrap(
                        ui.output_text_verbatim("input_validation_feedback1", 
                                                "Please ensure to have filled all details correctly in order to generate a prediction.",
                                                style="text-align: center; margin-top:10px; color:red"
                                                ),
                        ui.output_text_verbatim("input_validation_feedback2", 
                                                "No cooperation can be set up because of commiting extreme war crimes.",
                                                style="text-align: center; margin-top:10px; color:red"
                                                ),
                        ui.output_text_verbatim("input_validation_feedback3",
                                                "Please ensure to have filled all details correctly in order to generate a prediction. Total Cost and the Maximim EU contribution for the project cannot be 0.",
                                                style="text-align: center; margin-top:10px; color:red"
                                                ),
                        ui.output_text_verbatim("input_validation_feedback4",
                                                "Please ensure to have filled all details correctly in order to generate a prediction. Duration of the project (in days) of and numbers of organizations participating cannot be 0.",
                                                style="text-align: center; margin-top:10px; color:red"
                                                ),
                    )
        ),
        
        # tab Information 
        ui.nav_panel("How the Prediction",
                    ui.h4("How the Prediction is Made:", style="text-align: center; font-weight: bold;"),
                    ui.div(
                        ui.p("This application predicts the startup delay of projects funded by the Europe Horizon 2021-2027 programme. The prediction is based on various project parameters such as total cost, maximum EU contribution, number of participating organizations, duration, and the legal basis of the project."),
                        ui.p("The prediction is based on a machine learning model trained on historical project data. The model considers factors such as the total cost of the project, the maximum EU contribution, and the number of participating organizations."),
                        style="text-align: center; margin-top: 20px;"
                    )
                ),
        )

)

# defining steps for the server
def server(input, output, session):
    
    # validation of user input 
    iv = InputValidator()      

    iv.add_rule("countr_coor", check.custom(input_field_filled))
    iv.add_rule("pillar", check.custom(input_field_filled))
    iv.add_rule("number_org", check.custom(input_field_filled))
    iv.add_rule("ec_max_contribution", check.custom(input_field_filled))
    iv.add_rule("total_cost", check.custom(input_field_filled))
    iv.add_rule("duration", check.custom(input_field_filled))

    iv.add_rule("duration", check.custom(input_field_not_integer))
    iv.add_rule("number_org", check.custom(input_field_not_integer))

    iv.add_rule("duration", check.custom(input_field_not_zero))
    iv.add_rule("number_org", check.custom(input_field_not_zero))
    iv.add_rule("total_cost", check.custom(input_field_not_zero))
    iv.add_rule("ec_max_contribution", check.custom(input_field_not_zero))

    iv.enable()

    # storing prediction result in a reactive value to avoid re-computation
    prediction_result = reactive.Value("")

    # processing input after user clicks "submit"
    @reactive.event(input.submit):
    def get_prediction():
        try: 
            # user input saved as variables
            total_cost = input.totalCost()
            ec_max_contribution = input.ecMaxContribution()
            pillar = input.legalBasis()
            country_coor = input.countryCoor()
            duration = input.duration()
            number_org = input.numberOrg()
            
            # data validation 
            if total_cost is None or ec_max_contribution is None:
                return "input_validation_feedback3"
            if pillar is None or country_coor is None:
                return "input_validation_feedback1"
                
            if country_coor in ["IL", "Israel"]:
                return "input_validation_feedback2"
                
            if duration is None or duration != int(duration):
                return "input_validation_feedback4"
            
            if number_org is None or number_org != int(number_org):
                    return "input_validation_feedback4"

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
            api_url = "https://8icrl41qp8.execute-api.eu-west-3.amazonaws.com/prod/predict"
            post_request = httpx.post(api_url, json=data_to_api, timeout=45.0)

            # check for HTTP errors and raise exception if any and print
            post_request.raise_for_status()
            result = post_request.json()
            print("API response JSON:", result) 

            # receiving prediction - json format
            prediction = result.get("prediction")
            if prediction is None:
                return prediction_result.set(f"No prediction was returned. Response: {result}")

            else:
                prediction_result.set(f"{prediction}")
    
        except httpx.HTTPStatusError as status_error:
            # handle HTTP errors separately to show status code and message
            return prediction_result.set(f"HTTP error occurred: {status_error.response.status_code} - {status_error.response.text}")

        except httpx.RequestError as request_error:
            # handle request errors (e.g., connection issues)
            return prediction_result.set(f"Request error occurred: {str(request_error)}")

        except Exception as e:
            # general catch-all for other exceptions
            return prediction_result.set(f"Unexpected error: {str(e)}")
        
    # calling get_predition function to display prediction on UI
    @output
    @render.text
    def prediction_output():
        result = prediction_result()

        # If the result is an error message, show it directly
        if isinstance(result, str) and not result.startswith("Predicted"):
            return result

        # Compose a summary with input values + prediction
        return f"""
    Prediction Based
    -------------------
    Europe Horizon Pillar: {input.legalBasis()}
    Country Coordinator:   {input.countryCoor()}
    Total Cost:            {input.totalCost()}
    EU Max Contribution:   {input.ecMaxContribution()}
    Duration (days):       {input.duration()}
    Number of Orgs:        {input.numberOrg()}

    Predicted Startup Delay (in days): 

    {result}
    """
    

app = App(app_ui, server)