from shiny import App, render, ui
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load datasets
datasets = {
    "iris": sns.load_dataset("iris"),
    "mtcars": sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data,
    "trees": sm.datasets.get_rdataset("trees", "datasets", cache=True).data
}

# Define UI
app_ui = ui.page_fluid(
    ui.panel_title("Eighth app ..."),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("data1", "Select the Dataset of your choice", {k: k for k in datasets.keys()}),
            ui.output_ui("vx"),
            ui.output_ui("vy")
        ),
        ui.output_plot("plot")
    )
)

# Define Server
def server(input, output, session):
    def get_variable_choices():
        return list(datasets[input.data1()].columns)
    
    @output
    @render.ui
    def vx():
        return ui.input_select("variablex", "Select the First (X) variable", get_variable_choices())
    
    @output
    @render.ui
    def vy():
        return ui.input_select("variabley", "Select the Second (Y) variable", get_variable_choices())
    
    @output
    @render.plot
    def plot():
        df = datasets[input.data1()]
        x_var, y_var = input.variablex(), input.variabley()
        
        if x_var and y_var:
            x = df[x_var]
            y = df[y_var]
            
            fig, ax = plt.subplots()
            ax.scatter(x, y, label="Data Points")
            
            # Perform linear regression
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            ax.plot(x, model.predict(X), color='red', label="Regression Line")
            
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
            ax.set_title("Linear Regression")
            ax.legend()
            
            return fig

# Create the Shiny app
app = App(app_ui, server)