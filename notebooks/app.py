from shiny import App, ui, render, reactive
import requests

mapping = {
        "HORIZON.1.1": "Pillar 1 - European Research Council (ERC)",
        "HORIZON.1.2": "Pillar 1 - Marie Sklodowska-Curie Actions (MSCA)",
        "HORIZON.1.3": "Pillar 1 - Research infrastructures",
        "HORIZON.2.1": "Pillar 2 - Health",
        "HORIZON.2.2": "Pillar 2 - Culture, creativity and inclusive society",
        "HORIZON.2.3": "Pillar 2 - Civil Security for Society",
        "HORIZON.2.4": "Pillar 2 - Digital, Industry and Space",
        "HORIZON.2.5": "Pillar 2 - Climate, Energy and Mobility",
        "HORIZON.2.6": "Pillar 2 - Food, Bioeconomy Natural Resources, Agriculture and Environment",
        "HORIZON.3.1": "Pillar 3 - The European Innovation Council (EIC)",
        "HORIZON.3.2": "Pillar 3 - European innovation ecosystems",
        "HORIZON.3.3": "Pillar 3 - Cross-cutting call topics",
        "EURATOM2027": "EURATOM2027",
        "EURATOM.1.1": "Improve and support nuclear safety...",
        "EURATOM.1.2": "Maintain and further develop expertise...",
        "EURATOM.1.3": "Foster the development of fusion energy...",
}

app_ui = ui.page_fluid(
    ui.panel_title("Prediction of Startup Delay - Projects Europe Horizon 2021-2027"),
    ui.input_select("legalBasis", "Select Europe Horizon Pillar for your project", {k: k for k in mapping.keys()}),
    ui.input_select("countryCoor", "Select Country of Coordinating Organization of Your Project"),
    ui.input_action_button("submit", "Submit"),
    ui.output_text_verbatim("prediction_output")
)

def server(input, output, session):
    @reactive.event(input.submit)
    def get_prediction():
        return "Waiting for API integration..."

    @output
    @render.text
    def prediction_output():
        return get_prediction()

app = App(app_ui, server)
