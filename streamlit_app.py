import streamlit as st
# https://discuss.streamlit.io/t/streamlit-option-menu-is-a-simple-streamlit-component-that-allows-users-to-select-a-single-item-from-a-list-of-options-in-a-menu
# https://icons.getbootstrap.com/
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
import os
from streamlit_components import data_extractor, setting_setup, search_ai, logo_heading, usage_report, prompt_edit, set_png_as_page_bg

load_dotenv()

APP_TITLE = os.getenv('APP_TITLE', 'Document AI 🤖')

st.set_page_config(page_title=APP_TITLE, layout='wide')

admin_pass = "123"

styles = {
    "container": {"margin": "0px !important", "padding": "0!important", "align-items": "stretch", "background-color": "#FFF1DC"},
    "icon": {"color": "white", "font-size": "21px"},
    "nav-link": {"font-size": "12px", "text-align": "left", "margin":"0px", "color": "white",
                 "--hover-color": "#e0c5ff", "background-color": "#9385E0"},
    "nav-link-selected": {"background-color": "#6A46BF", "font-size": "15px",
                          "font-weight": "bold", "color": "white", "border":"1px solid #b899e4"},
}

styles_main = {
    "menu-title":{"font-size": "22px", "text-align": "left", "margin":"0px", "font-weight": "bold",
                   "padding": "0!important", "color": "#8238BB"},
    "menu-icon":{"color": "#8238BB", "font-size": "25px"},
    "container": {"margin": "1px !important", "padding": "1!important", "align-items": "stretch", "background-color": "#ece0fa"},
    "icon": {"color": "black", "font-size": "15px"},
    "nav-link": {"font-size": "12px", "text-align": "left", "margin":"0px", "--hover-color": "#e0c5ff"},
    "nav-link-selected": {"background-color": "#6A46BF", "font-size": "15px",
                          "font-weight": "bold", "color": "white", "border":"1px solid #b899e4"},
}

if "admin_pass" in st.session_state and st.session_state.admin_pass == admin_pass:
    menu = {
        'title': APP_TITLE,
        'items': {
            'Admin Pannel' : {
                        'action': None, 'item_icon': 'database-fill-gear',
                        'submenu': {
                            'title': "",
                            'items': {
                                'Usage Report & Feedback': {'action': usage_report, 'item_icon': 'clipboard-data','submenu': None},
                                'RAG Setup' : {'action': data_extractor, 'item_icon': 'database-fill-gear' , 'submenu': None},
                                'Edit Prompt': {'action': prompt_edit, 'item_icon': 'text-indent-left', 'submenu': None}
                            },
                            'menu_icon': 'None',
                            'default_index': 0,
                            'with_view_panel': 'main',
                            'orientation': 'horizontal',
                            'styles': styles
                        }
                    },

            'AI Assistant' : {
                        'action': None, 'item_icon': 'search',
                        'submenu': {
                            'title': "",
                            'items': {
                                'AI Assistant' : {'action': search_ai, 'item_icon': 'search', 'submenu': None}
                            },
                            'menu_icon': 'None',
                            'default_index': 0,
                            'with_view_panel': 'main',
                            'orientation': 'horizontal',
                            'styles': styles
                        }
                    },
            'Settings' : {
                'action': None, 'item_icon': 'gear',
                'submenu': {
                    'title': None,
                    'items': {
                        'App Settings' : {'action':setting_setup, 'item_icon': 'gear', 'submenu': None},
                    },
                    'menu_icon': None,
                    'default_index': 0,
                    'with_view_panel': 'main',
                    'orientation': 'horizontal',
                    'styles': styles
                }
            }
        },
        'menu_icon': 'cast',
        'default_index': 0,
        'with_view_panel': 'sidebar',
        'orientation': 'vertical',
        'styles': styles_main
    }
else:
    menu = {
        'title': APP_TITLE,
        'items': {
            'AI Assistant': {
                'action': None, 'item_icon': 'search',
                'submenu': {
                    'title': "",
                    'items': {
                        'AI Assistant': {'action': search_ai, 'item_icon': 'search', 'submenu': None}
                    },
                    'menu_icon': 'None',
                    'default_index': 0,
                    'with_view_panel': 'main',
                    'orientation': 'horizontal',
                    'styles': styles
                }
            },
            'Settings': {
                'action': None, 'item_icon': 'gear',
                'submenu': {
                    'title': None,
                    'items': {
                        'App Settings': {'action': setting_setup, 'item_icon': 'gear', 'submenu': None},
                    },
                    'menu_icon': None,
                    'default_index': 0,
                    'with_view_panel': 'main',
                    'orientation': 'horizontal',
                    'styles': styles
                }
            }
        },
        'menu_icon': 'cast',
        'default_index': 0,
        'with_view_panel': 'sidebar',
        'orientation': 'vertical',
        'styles': styles_main
    }

def show_menu(menu):
    def _get_options(menu):
        options = list(menu['items'].keys())
        return options

    def _get_icons(menu):
        icons = [v['item_icon'] for _k, v in menu['items'].items()]
        return icons

    kwargs = {
        'menu_title': menu['title'] ,
        'options': _get_options(menu),
        'icons': _get_icons(menu),
        'menu_icon': menu['menu_icon'],
        'default_index': menu['default_index'],
        'orientation': menu['orientation'],
        'styles': menu['styles']
    }

    with_view_panel = menu['with_view_panel']
    if with_view_panel == 'sidebar':
        with st.sidebar:
            menu_selection = option_menu(**kwargs)
    elif with_view_panel == 'main':
        SHOW_TITLE_IMAGE = bool(os.getenv('SHOW_TITLE_IMAGE', False))
        if SHOW_TITLE_IMAGE:
            logo_heading()
        set_png_as_page_bg()
        menu_selection = option_menu(**kwargs)
    else:
        raise ValueError(f"Unknown view panel value: {with_view_panel}. Must be 'sidebar' or 'main'.")

    if menu['items'][menu_selection]['submenu']:
        show_menu(menu['items'][menu_selection]['submenu'])

    if menu['items'][menu_selection]['action']:
        menu['items'][menu_selection]['action']()

show_menu(menu)
