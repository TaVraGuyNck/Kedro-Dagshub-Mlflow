# two-Column Layou

        # Left column: Paragraph/text content
        ui.column(6,
            ui.div(
                ui.input_text_area(cols=50, rows=8, id="descrption", label="Description",
                                   placeholder="This application predicts the startup delay of projects funded by the Europe" \
                                   " Horizon 2021-2027 programme. The prediction is based on various project parameters such as total cost," \
                                   " maximum EU contribution, number of participating organizations, duration, and the legal basis of the project. " \
                                   "Please fill in the details on the right to get a prediction."
                                      "The prediction is based on a machine learning model trained on historical project data. " \
                                   "The model considers factors such as the total cost of the project, the maximum EU contribution, " \
                                    ),
                style="padding-right: 20px;"
            ),

        # middle part
            
            )
        ),

        # right column - data input fields, title, submit button
        ui.column(6,
            ui.div(
              
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
                 ui.layout_column_wrap(
                    
                    )
        ),

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