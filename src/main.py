import dearpygui.dearpygui as dpg

dpg.create_context()
#function for resize handler
def app_resize(sender, app_data):
    # Get viewport size
    width = app_data[0]
    height = app_data[1]
    # Set new input size
    dpg.set_item_width("Main text input", int(width * 2 / 3))
    dpg.set_item_height("Main text input", int(height-75))
    #Set new AI window size
    dpg.set_item_width("AI window", int(width/3 - 60))
    dpg.set_item_height("AI window", int(height-75))
    dpg.set_item_pos("AI window", (width/3*2+25,27))
    dpg.set_item_width("AI text request", dpg.get_item_width("AI window"))
    dpg.set_item_pos("AI text request", (0, dpg.get_item_height("AI window")-27))

with dpg.window(tag="Main window",width=500, height=300):

    with dpg.menu_bar():
        dpg.add_button(tag="saveFile", label="Save")
        dpg.add_button(tag="ImportFile", label="Import")
    dpg.add_input_text(tag="Main text input", multiline=True, width=600, height=500)

    with dpg.window(tag="AI window", no_title_bar=True, no_move=True, no_resize=True):
        dpg.add_text("AI chat will be here")
        dpg.add_input_text(tag="AI text request", pos=(0, 495),default_value="Type your question here")

    #handler for auto size
    dpg.set_viewport_resize_callback(app_resize)

dpg.create_viewport(title='LexifyAI', width=800, height=600)
dpg.setup_dearpygui()
#dpg.show_font_manager()
dpg.show_viewport()
dpg.set_primary_window("Main window", True)
dpg.start_dearpygui()
dpg.destroy_context()