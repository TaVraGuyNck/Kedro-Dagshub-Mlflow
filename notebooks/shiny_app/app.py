from shiny import App, ui, render, reactive
from datetime import datetime
import httpx
from pathlib import Path

# to ensure connection to http api
api_uri = "https://x58fyx1367.execute-api.eu-west-3.amazonaws.com/prod/predict"


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
    " ": " ",
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
        ui.h2("Real-time Prediction/Estimation of Startup Delay - Projects Europe Horizon 2021-2027", style="text-align:center; font-weight: bold;"),
        ui.h6("Mean Absolute Error (MAE) of +/- 69 days", style="text-align:center; font-weight: bold"),
        style="margin-bottom: 30px;"
    ),
    ui.navset_tab(
        ui.nav_panel(
            "Prediction Start-up Delay",
            ui.h4("Please Enter Following Project Details:", style="text-align: center; font-weight: bold;"),

            ui.layout_columns(
                ui.card(
                    ui.card_header("Under which Horizon Europe Pillar falls the Project?"),
                    ui.input_select("pillar", "", choices=mapping)
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
                     ui.h4("Introduction", style="text-align: center; font-weight: bold;"), 
                     ui.div(
                         ui.p(
                             """A good start is half the battle. This project aims to predict/estimate the start-up delay of projects under the Horizon Europe 2021–2027 program.
                             Start-up delay refers to the time gap between the date of the signature of the grant agreement and the actual start date of the project.
                             The main objective is to help program administrators identify projects at risk of significant start-up delays, so extra support can be provided early on.
                             In this application, a real-time prediction/estimation of the startup-delay can be obtained by entering the details of a single project.
                              It is however to be taken into account that our model has an average error in prediction of +/- 69 days. Predictions/estimations are therewith to be considered more as a probable indication, not as a hard value.  """,
                             style="text-align: justify; font-size: 16px; margin: 20px auto; max-width: 900px;"
                             )
                     ),
                     ui.h4("Explanation on the predicting model and the accuracy of its predictions", style="text-align: center; font-weight: bold;"),
                     ui.h6("Catboost model:", style="text-align: center; font-weight: bold;"),
                     ui.div(
                         ui.p(
                             """Model used is a Catboost regression model, which is a gradient-boosted decision tree algorithm (ensemble method).
                              It is well-suited for datasets with a mix of numerical and categorical features.
                              We experimented as well with a XGBoost model and a linear regression model. 
                              Based on comparison of the Mean Absolute Error (MAE) Catboost came out as the model with the lowest MAE. 
                              NOTE: Even with lowest MAE, the MAE for predictions of the startup-delay by our Catboost model is still
                              +/- 69 days. Please see "Model Performance" hereunder for additional info. """,
                              style="text-align: justify; font-size: 16px; margin: 20px auto; max-width: 900px;"),
                     ),
                     ui.h6("Data used to train the model and its preprocessing:", style="text-align: center; font-weight: bold;"),
                     ui.div(
                         ui.p(
                             """We used the data generated and published by Europe Horizon 2021-2027, which includes all projects which have started
                              since the beginning of Europe Horizon 2021-2027. (Including projects which already ended in the meantime.)
                              For the preprocessing of the data, before using them to train the model: missing values were imputed (median for numeric, "missing" for categorical),
                              After splitting of the data into a training and a testing set (80%-20%) outliers were removed using Isolation Forest. 
                              Categorical data was left as strings (not encoded), as Catboost handles it natively.""",
                              style="text-align: justify; font-size: 16px; margin: 20px auto; max-width: 900px;"),
                     ),

                     ui.h6("Input features used:", style="text-align: center; font-weight: bold;"),
                     ui.div(
                         ui.p(
                             """Features used to generate the prediction are: Funding Scheme of the project, Duration of the project (in days), 
                             EU Horizon Pillar under which the project falls, Total Cost of the project, maximum contribution to this cost by EC, 
                             Country of the Coordinating Organization of the project, number of organisations participating to the project. Based on those, 
                             2 additional features are created: ratio between Total Cost of the project and the maximum contribution by the EC, and a dummy
                              variable in case of a Total Cost of 0 (used during training). """,
                             style="text-align: justify; font-size: 16px; margin: 20px auto; max-width: 900px;"),
                     ),

                     ui.h6("Model performance:", style="text-align: center; font-weight: bold;"),
                     ui.div(
                         ui.p("""On the test data set, we obtained following evaluation metrics for our model:  
                              R² = 0.365 (Coefficient of Determination: evaluates that our model explains ~36.5% of 
                              the variance of startup-delay (with the other ~73,5% currently not being explained by our model.)

                              MAE = 69.075 days (Mean Absolute Error: evaluates that the predictions of our model currently are off 
                              by +/- 69 days).

                              RMSE = 95.267 days (Root Mean Squared Error, an evaluation metric which is more sensitive for large 
                              errors than MAE: evaluates that the predictions of our model, while giving more weight to large errors, 
                              are currently off by +/- 95 days).
                              """,
                             style="white-space: pre-wrap; text-align: justify; font-size: 16px; margin: 20px auto; max-width: 900px;"
                         )
                     ),
                     ui.h6("Final Note:", style="text-align: center; font-weight: bold;"),
                     ui.div(
                         ui.p("""It is important to take into account that at this stage, the "prediction" of the startup-delay 
                              by our model only has an accuracy of 69 days more or less than the amount of days returned as result. 
                              There is still a lot of variance in the startup-delay over projects, which is not taken into account by our model.
                              More analysis and experimenting is needed, in order to improve the accuracy rate of our model's predictions.""",
                              style="text-align: justify; font-size: 16px; margin: 20px auto; max-width: 900px;"),
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
        if not input.countryCoor() or input.countryCoor() in [" ", "", "  "]:
            return "Please select the country of the Coordinating Organization."
        if input.pillar() is None or input.pillar() in [" ", "", "  "]:
            return "Please select the correct pillar to which the project belongs."
        if not input.fundingScheme() or input.fundingScheme() in [" ", "", "  "]:
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
                return f"{prediction}"

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
            {
                "style": """
                white-space: pre-wrap;
                border: 2px solid #ccc;
                padding: 16px;
                margin-top: 20px;
                font-family: monospace;
                background-color: #f9f9f9;
                width: 100%;
                box-sizing: border-box;
                """
            },
            f"""

OVERVIEW OF DETAILS ON PROJECT AS PROVIDED BY USER (used to generate prediction/estimation):
__________________________________________________________________________________________

Europe Horizon Pillar:--------------------------------------------- {input.pillar()}
Country of Coordinating Organization:----------------------------- {input.countryCoor()}
Total Cost:-------------------------------------------------------- {input.totalCost()}
EC Max Contribution:------------------------------------------------{input.ecMaxContribution()}
Duration (in days):------------------------------------------------{input.duration()}
Number of Participating Organizations:-----------------------------{input.numberOrg()}
Funding Scheme:-----------------------------------------------------{input.fundingScheme()}
__________________________________________________________________________________________

RESULT - PREDICTION/ESTIMATION OF STARTUP-DELAY IN DAYS BY THIS MODEL:

{result} 
(+/- 69 days average error)
""",
ui.tags.div(
        {"style": "color: red; font-weight: bold;"},
        """IMPORTANT NOTE: 
Predictions/estimations generated by our model have an average error of ±69 days (MAE).

For more information on the model used to generate these predictions/estimations, how it 
was developed and what sources of error should be taken into account, please refer 
to the Information tab of this application."""
    ))
 
       
app = App(app_ui, server)