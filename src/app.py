import dearpygui.dearpygui as dpg

dpg.create_context()

def print_me(sender):
    print(f"Menu Item: {sender}")

# Создание меню
with dpg.viewport_menu_bar():
    with dpg.menu(label="File"):
        dpg.add_menu_item(label="Save", callback=print_me)
        dpg.add_menu_item(label="Save As", callback=print_me)

        with dpg.menu(label="Settings"):
            dpg.add_menu_item(label="Setting 1", callback=print_me, check=True)
            dpg.add_menu_item(label="Setting 2", callback=print_me)

    dpg.add_menu_item(label="Help", callback=print_me)

# Callback функция для обновления номеров строк
def update_line_numbers(sender, app_data):
    text = dpg.get_value(sender)  # Получаем текущий текст из редактора кода
    lines = text.count('\n') + 1  # Считаем количество строк
    line_numbers = '\n'.join(str(i + 1) for i in range(lines))  # Формируем нумерацию строк
    dpg.set_value("line_numbers", line_numbers)  # Обновляем текст с номерами строк

# Создаем главное окно с таблицей для разметки
with dpg.window(label="Main window", width=800, height=600):
    # Создаем таблицу с 2 колонками и 2 строками для разметки
    with dpg.table(header_row=False, borders_innerH=True, borders_innerV=True, borders_outerH=True, borders_outerV=True):
        # Определяем две колонки: одна для структуры проекта, другая для редактора
        dpg.add_table_column(init_width_or_weight=1)  # Левая колонка (структура проекта)
        dpg.add_table_column(init_width_or_weight=3)  # Правая колонка (редактор кода)

        # Строка 1: основная рабочая область (структура проекта и редактор кода)
        with dpg.table_row():
            # Левая часть — структура проекта
            with dpg.child_window(width=-1, height=400):  # Отключаем прокрутку
                dpg.add_text("Project Structure")
                dpg.add_tree_node(label="It will be Project Root and etc..")

            # Правая часть — редактор кода с включенной прокруткой
            with dpg.child_window(width=-1, height=400):  # Прокрутка разрешена только здесь
                with dpg.group(horizontal=True):  # Используем горизонтальную группу для разметки
                    # Нумерация строк
                    dpg.add_text("1", tag="line_numbers", wrap=50, indent=10)  # Текст для номеров строк

                    # Редактор кода
                    dpg.add_input_text(tag="code_editor", multiline=True, width=-1, height=400,
                                       #default_value="def main():\n    print('Hello, World!')",
                                       callback=update_line_numbers, on_enter=False)

        # Строка 2: объединяем две колонки для нижней части (span columns)
        with dpg.table_row():
            # Здесь мы используем одну объединенную ячейку
            with dpg.child_window(width=-1, height=150):  # Отключаем прокрутку
                dpg.add_text("Folder Explorer")
                dpg.add_input_text(multiline=True, height=-1, width=-1, default_value="repositories and files of your PC")

# Запускаем рендеринг окна
dpg.create_viewport(title='LexifyAI', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
