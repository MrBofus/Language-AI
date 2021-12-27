import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, State, Input
import pandas as pd
import torch
from tqdm import trange
from transformers import GPT2Tokenizer
import torch.nn.functional as F
from dash import dash_table

################################################################################
################################################################################

temp = 0.95
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = torch.load('model.pt')


user_key = 'access_token'


NAME = str('Type some prompts here! The longer the prompts, the better the story will sound')
columns = [NAME]
table = dash_table.DataTable(columns=[{"name": column, "id": column} for column in columns], data=[], id="table")


################################################################################
################################################################################

def generate(
    model,
    tokenizer,
    prompt,
    entry_count=10,
    entry_length=45, #maximum number of words
    top_p=0.8,
    temperature=temp,
):

    model.eval()

    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():

        for entry_idx in trange(entry_count):

            entry_finished = False

            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            for i in range(entry_length):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token in tokenizer.encode("<|endoftext|>"):
                    entry_finished = True

                if entry_finished:

                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generated_list.append(output_text)
                    break
            
            if not entry_finished:
              output_list = list(generated.squeeze().numpy())
              output_text = f"{tokenizer.decode(output_list)}<|endoftext|>" 
              generated_list.append(output_text)
                
    return generated_list

def text_generation(test_data):
    generated_lyrics = []
    for i in range(len(test_data)):
        x = generate(model.to('cpu'), tokenizer, test_data[NAME][i], entry_count=1)
        generated_lyrics.append(x)
    return generated_lyrics

################################################################################
################################################################################

app = dash.Dash(__name__)

#app.layout = html.Div(
#    [html.Div(
#        className = "app-header",
#        children = [html.Div('Plotly Dash', className="app-header--title")]
#        )] +
#    [html.Div(
#        children=html.Div([
#            html.H5('Overview'),
#            html.Div('''
#                This is an example of a simple Dash app with
#                local, customized CSS.
#            ''')
#        ]))] +
#    [html.H6("Title")] +
#    [dcc.Input(id=column, value=column) for column in columns] +
#    [html.Button("Save", id="save"), dcc.Store(id="cache", data=[]), table] +
#    [html.Div(id='my-output', style={'whiteSpace': 'pre-line'})]
#    )

app.layout = html.Div([
            html.H1(children="Hello world!",className="hello",style={
    'color':'#00361c','text-align':'center'}),
    
html.Div([[dcc.Input(id=column, value=column) for column in columns] +
[html.Button("Save", id="save"), dcc.Store(id="cache", data=[]), table] +
[html.Div(id='my-output', style={'whiteSpace': 'pre-line'})]]),

        html.Div([
                html.Div(
                        children="Block 1",className="box1",
                        style={
                        'backgroundColor':'darkslategray',
                        'color':'lightsteelblue',
                        'height':'100px',
                        'margin-left':'10px',
                        'width':'45%',
                        'text-align':'center',
                        'display':'inline-block'
                        }),
            
                 html.Div(
                        children="Block 2",className="box2",
                        style={
                        'backgroundColor':'darkslategray',
                        'color':'lightsteelblue',
                        'height':'100px',
                        'margin-left':'10px',
                        'text-align':'center',
                        'width':'40%',
                        'display':'inline-block'
                        })
             ])
        
            
      ])


@app.callback(Output("table", "data"),
              Output(component_id='my-output', component_property='children'),
              Input("save", "n_clicks"), 
              [State("table", "data")] + [State(column, "value") for column in columns]
              )

################################################################################
################################################################################
    
def update_output(clicks, data, *args):
    if clicks is not None:
        data.append({columns[i]: arg for i, arg in enumerate(list(args))})

        if (int(clicks) % 3 == 0):
            start_str = pd.DataFrame(data)
            print(start_str)
            data = []
            generated_story = text_generation(start_str)

            i = 0
            generated_story_str = '\t'

            while i < len(generated_story):
                generated_story_str = generated_story_str + ' ' + str(generated_story[i])
                i += 1

            generated_story = generated_story_str

            generated_story = generated_story.replace('[\'', '')
            generated_story = generated_story.replace('[\"','')
            generated_story = generated_story.replace('\']', '')
            generated_story = generated_story.replace('\"]','\"')
            generated_story = generated_story.replace('\\xa0', '')
            generated_story = generated_story.replace('<|endoftext|>','.')
            generated_story = generated_story.replace('.,','.')
            generated_story = generated_story.replace(',.','.')
            generated_story = generated_story.replace('..','.')
            generated_story = generated_story.replace('      DRACULA     CHAPTER I  JONATHAN HARKERS JOURN','')
            generated_story = generated_story.replace('\\n\\n', ' ')
            generated_story = generated_story.replace('\\n','')
            generated_story = generated_story.replace('.\".', '.\"')

            print('\n\n' + generated_story + '\n\n')
                
            return data, str('\n\n' + generated_story + '\n\n')
        return data, ''
    ################################################################################
    ################################################################################


if __name__ == '__main__': 
    app.run_server()