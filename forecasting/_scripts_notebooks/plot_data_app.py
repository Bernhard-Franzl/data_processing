import dash
from dash import Dash, html, dcc
#from components import page_header

app = Dash(name=__name__,
           use_pages=True,
           pages_folder='_data_viewer/pages',
           suppress_callback_exceptions=True
           )

app.layout = html.Div([
    #page_header.layout(),
    html.H1('Data Viewer'),
    html.Div([
        html.Div(
            dcc.Link(f"{page['name']} - {page['path']}", href=page["relative_path"])
        ) for page in dash.page_registry.values()
    ]),
    dash.page_container
    ],
    style={"margin":"0px",
           "padding":"0px"}
)

if __name__ == '__main__':
    app.run(debug=True)