import glob
import os
import logging

import pandas as pd
import gradio as gr
from gradio.themes.utils.sizes import text_md

from content import (HEADER_MARKDOWN, LEADERBOARD_TAB_TITLE_MARKDOWN, SUBMISSION_TAB_TITLE_MARKDOWN,
                     )

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import time
import gradio as gr

from huggingface_hub import HfApi, snapshot_download


JSON_DATASET_DIR = Path("../json_dataset")
JSON_DATASET_DIR.mkdir(parents=True, exist_ok=True)

JSON_DATASET_PATH = JSON_DATASET_DIR / f"train-{uuid4()}.json"


api = HfApi()

ORG= "CZLC"
REPO = f"{ORG}/LLM_benchmark_data"

def greet(name: str) -> str:
    return "Hello " + name + "!"


DATASET_VERSIONS = ['dev-set-1', 'dev-set-2']

HF_TOKEN = os.environ.get("HF_TOKEN")



class LeaderboardServer:
    def __init__(self, server_address):
        self.server_address = server_address
        self.repo_type = "dataset"
        self.local_leaderboard = snapshot_download(self.server_address,repo_type=self.repo_type, token=HF_TOKEN)

    def on_submit(self):
        self.local_leaderboard = snapshot_download(self.server_address,repo_type=self.repo_type, token=HF_TOKEN)

    def get_leaderboard(self):
        results = []
        for submission in glob.glob(os.path.join(self.local_leaderboard, "../data") + "/*.json"):
            data = json.load(open(submission))
            submission_id = data["metadata"]["model_description"]
            local_results = {group: data["results"][group]['acc'] for group in data['results']}
            local_results["submission_id"] = submission_id
            results.append(local_results)
        dataframe = pd.DataFrame.from_records(results)
        return dataframe

    def save_json(self,file) -> None:
        filename = os.path.basename(file)
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=f"data/{filename}",
            repo_id=self.server_address,
            repo_type=self.repo_type,
            token=HF_TOKEN,
        )



leaderboard_server =  LeaderboardServer(REPO)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


LEADERBOARD_TYPES = ['LLM',]
MAX_SUBMISSIONS_PER_24H = 2
# DATASET_VERSIONS = ['dev-set-1', 'dev-set-2']
# CHALLENGE_NAME = 'NOTSOFAR1'


if __name__ == '__main__':
    with (gr.Blocks(theme=gr.themes.Soft(text_size=text_md), css="footer {visibility: hidden}") as main):
        app_state = gr.State({})
        # with gr.Row():
        #     greet_name = gr.Textbox(label="Name")
        #     greet_output = gr.Textbox(label="Greetings")
        # greet_btn = gr.Button("Greet")
        # greet_btn.click(fn=greet, inputs=greet_name, outputs=greet_output).success(
        #     fn=save_json,
        #     inputs=[greet_name, greet_output],
        #     outputs=None,
        # )

        with gr.Row():
            with gr.Row():
                gr.Markdown(HEADER_MARKDOWN)

        with gr.Row():

            # Leaderboards Tab #
            ####################
            def populate_leaderboard(leaderboard_type, dataset_version):
                gr.Info('Loading leaderboard...')
                time.sleep(1)
                leaderboard_df = leaderboard_server.get_leaderboard()
                # leaderboard_df = lb_server.get_leaderboard(
                #     submission_type=leaderboard_type, dataset_version=dataset_version)
                # if leaderboard_df.empty:
                return leaderboard_df
                # return leaderboard_df


            def create_leaderboard_tab(tab_name: str, idx: int, dataset_version_dropdown: gr.Dropdown):
                # dataset_version = dataset_version_dropdown.value
                print(f'Creating tab for {tab_name}, idx={idx}, dataset_version={dataset_version_dropdown}')
                with gr.Tab(id=tab_name, label=tab_name) as leaderboard_tab:
                    leaderboard_table = gr.DataFrame(populate_leaderboard(tab_name, None)) if idx == 0 \
                        else gr.DataFrame(pd.DataFrame(columns=['No submissions yet']))
                    leaderboard_tab.select(fn=populate_leaderboard,
                                           inputs=[gr.Text(tab_name, visible=False)],
                                           outputs=[leaderboard_table])
                    return leaderboard_table

            def on_dropdown_change():
                first_tab_name = LEADERBOARD_TYPES[0]
                leaderboard_server.on_submit()

                return gr.Tabs(selected=first_tab_name), populate_leaderboard(first_tab_name, None)


            with gr.Tab('Leaderboards') as leaderboards_tab:
                with gr.Row():
                    gr.Markdown(LEADERBOARD_TAB_TITLE_MARKDOWN)
                # with gr.Row():
                #     with gr.Column():
                #         dataset_version_drop = gr.Dropdown(choices=DATASET_VERSIONS, multiselect=False,
                #                                            value=DATASET_VERSIONS[-1], label="Dataset",
                #                                            interactive=True)
                #     with gr.Column():
                #         gr.Markdown('')  # Empty column for spacing
                #     with gr.Column():
                #         gr.Markdown('')  # Empty column for spacing
                #     with gr.Column():
                #         gr.Markdown('')  # Empty column for spacing
                with gr.Row():
                    with gr.Tabs() as leaderboards_tabs:
                        leaderboard_tables_list = []
                        for leaderboard_idx, leaderboard_type in enumerate(LEADERBOARD_TYPES):
                            l_tab = create_leaderboard_tab(leaderboard_type, leaderboard_idx, None)
                            leaderboard_tables_list.append(l_tab)

                # dataset_version_drop.select(fn=on_dropdown_change, inputs=[dataset_version_drop],
                #                             outputs=[leaderboards_tabs, leaderboard_tables_list[0]])


            # Submission Tab #
            ##################
            with gr.Tab('Submission'):
                with gr.Column():
                    def on_submit_pressed():
                        return gr.update(value='Processing submission...', interactive=False)

                    def validate_submission_inputs(team_name, submission_zip, submission_type, token):
                        if not team_name or not submission_zip or not submission_type:
                            raise ValueError('Please fill in all fields')
                        if not os.path.exists(submission_zip):
                            raise ValueError('File does not exist')
                        # if not submission_zip.endswith('.zip'):
                        #     raise ValueError('File must be a zip')
                        # if not token:
                        #     raise ValueError('Please insert a valid Hugging Face token')

                    def process_submission(team_name, submission, submission_type, description,
                                           app_state, request: gr.Request):
                        logging.info(f'{team_name}: new submission for track: {submission_type}')
                        try:
                            token = app_state.get('hf_token')
                            validate_submission_inputs(team_name, submission, submission_type, token)
                        except ValueError as err:
                            gr.Warning(str(err))
                            return


                        # metadata = {'challenge_name': CHALLENGE_NAME,
                        #             "dataset_version": DATASET_VERSIONS[-1],
                        #             'team_name': team_name,
                        #             'submission_type': submission_type,
                        #             'description': description,
                        #             'token': token,
                        #             'file_name': os.path.basename(submission_zip),
                        #             'file_size_mb': os.path.getsize(submission_zip) / 1024 / 1024,
                        #             'ip': request.client.host}
                        leaderboard_server.save_json(submission)

                        try:
                            gr.Info('Processing submission...')
                            # response = lb_server.add_submission(token=token, file_path=submission_zip, metadata=metadata)
                            # if 'error' in response:
                            #     gr.Warning(f'Failed to process submission - {response["error"]}')
                            # else:
                            gr.Info('Done processing submission')
                        except Exception as e:
                            gr.Warning(f'Submission failed to upload - {e}')

                    def on_submit_done():
                        on_dropdown_change()
                        leaderboard_server.on_submit()
                        # leaderboard_tab.children[0] = gr.DataFrame(populate_leaderboard(None, None))
                        # leaderboard_tab.render()
                        return gr.update(value='Submit', interactive=True)

                    gr.Markdown(SUBMISSION_TAB_TITLE_MARKDOWN)
                    submission_team_name_tb = gr.Textbox(label='Team Name')
                    submission_file_path = gr.File(label='Upload your results', type='filepath')
                    submission_type_radio = gr.Radio(label='Submission Track', choices=LEADERBOARD_TYPES)
                    with gr.Row():
                        hf_token_tb = gr.Textbox(label='Token', type='password')
                        submissions_24h_txt = gr.Textbox(label='Submissions 24h', value='')
                    description_tb = gr.Textbox(label='Description', type='text')
                    submission_btn = gr.Button(value='Submit', interactive=True)
                    gr.Markdown('### * Please make sure you are using NOTSOFAR dev-set-2 for your submissions')

                    submission_btn.click(
                        fn=on_submit_pressed,
                        outputs=[submission_btn]
                    ).then(
                        fn=process_submission,
                        inputs=[submission_team_name_tb, submission_file_path,
                                submission_type_radio, description_tb, app_state]
                    ).then(
                        fn=on_submit_done,
                        outputs=[submission_btn]
                    ).then(
                        fn=on_dropdown_change,
                                        outputs=[leaderboards_tabs, leaderboard_tables_list[0]]
                    )

            # # My Submissions Tab #
            # ######################
            # with gr.Tab('My Submissions') as my_submissions_tab:
            #     def on_my_submissions_tab_select(app_state):
            #         hf_token = app_state.get('hf_token')
            #         if not hf_token:
            #             return pd.DataFrame(columns=['Please insert your Hugging Face token'])
            #         # submissions = lb_server.get_submissions_by_hf_token(hf_token=hf_token)
            #         # if submissions.empty:
            #         #     submissions = pd.DataFrame(columns=['No submissions yet'])
            #         # return submissions
            #
            #     gr.Markdown(MY_SUBMISSIONS_TAB_TITLE_MARKDOWN)
            #     my_submissions_table = gr.DataFrame()
            #
            #     my_submissions_tab.select(fn=on_my_submissions_tab_select, inputs=[app_state],
            #                               outputs=[my_submissions_table])
            #     my_submissions_token_tb = gr.Textbox(label='Token', type='password')

        def on_token_insert(hf_token, app_state):
            gr.Info(f'Verifying token...')

            submission_count = None
            # if hf_token:
                # submission_count = lb_server.get_submission_count_last_24_hours(hf_token=hf_token)

            if submission_count is None:
                # Invalid token
                app_state['hf_token'] = None
                submissions_24h_str = ''
                team_submissions_df = pd.DataFrame(columns=['Invalid Token'])
                gr.Warning('Invalid token')

            # else:
            #     app_state['hf_token'] = hf_token
            #     submissions_24h_str = f'{submission_count}/{MAX_SUBMISSIONS_PER_24H}'
            #     team_submissions_df = lb_server.get_submissions_by_hf_token(hf_token=hf_token)
            #     if team_submissions_df.empty:
            #         team_submissions_df = pd.DataFrame(columns=['No submissions yet'])
            #     gr.Info('Token verified!')

            return app_state, team_submissions_df, submissions_24h_str

        hf_token_tb.change(fn=on_token_insert, inputs=[hf_token_tb, app_state],
                           outputs=[app_state, submissions_24h_txt])
        # my_submissions_token_tb.change(fn=on_token_insert, inputs=[my_submissions_token_tb, app_state],
        #                                outputs=[app_state, my_submissions_table, submissions_24h_txt])

        main.launch()
