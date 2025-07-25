import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from typing import Iterator

import gradio as gr
import spaces
import torch
from aixblock_ml.model import AIxBlockMLBase
from huggingface_hub import HfFolder, login
from loguru import logger
from mcp.server.fastmcp import FastMCP
from transformers import (
    Gemma3nForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer
)

from function_ml import connect_project, download_dataset, upload_checkpoint
from logging_class import start_queue, write_log
from utils.chat_history import ChatHistoryManager
from model_docchat import docchat_answer
from config import constants
# from prompt import qa_without_context
# import gc

# ------------------------------------------------------------------------------
hf_token = os.getenv("HF_TOKEN", "hf_JWexoeUOV"+"wfTCxQdNTtLmpGDFIIIUuyeSn")
HfFolder.save_token(hf_token)


hf_access_token = "hf_JWexoeUOVwfT"+"CxQdNTtLmpGDFIIIUuyeSn"
login(token=hf_access_token)
CUDA_VISIBLE_DEVICES = []
for i in range(torch.cuda.device_count()):
    CUDA_VISIBLE_DEVICES.append(i)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    f"{i}" for i in range(len(CUDA_VISIBLE_DEVICES))
)
print(os.environ["CUDA_VISIBLE_DEVICES"])


HOST_NAME = os.environ.get("HOST_NAME", "https://dev-us-west-1.aixblock.io")
TYPE_ENV = os.environ.get("TYPE_ENV", "DETECTION")


mcp = FastMCP("aixblock-mcp")

CHANNEL_STATUS = {}
# Parameters for model demo
model_demo = None
tokenizer_demo = None
model_loaded_demo = False
# Parameters for model deployment
pipe_prediction = None
tokenizer = None
processor = None
model_predict = None


class MyModel(AIxBlockMLBase):

    @mcp.tool()
    def action(self, command, **kwargs):
        logger.info(f"Received command: {command} with args: {kwargs}")
        if command.lower() == "execute":
            _command = kwargs.get("shell", None)
            logger.info(f"Executing command: {_command}")
            subprocess.Popen(
                _command,
                shell=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
            )
            return {"message": "command completed successfully"}

        elif command.lower() == "train":

            model_id = kwargs.get("model_id", "google/gemma-3n-e4b-it")
            dataset_id = kwargs.get(
                "dataset_id", "autoprogrammer/Qwen2.5-Coder-7B-Instruct-codeguardplus"
            )

            push_to_hub = kwargs.get("push_to_hub", True)
            hf_model_id = kwargs.get("hf_model_id", "gemma-3n-e4b-it")
            push_to_hub_token = kwargs.get(
                "push_to_hub_token", "hf_JWexoeUOVwfT"+"CxQdNTtLmpGDFIIIUuyeSn"
            )
            framework = kwargs.get("framework", "huggingface")
            task = kwargs.get("task", "text-generation")
            prompt = kwargs.get("prompt", "")
            trainingArguments = kwargs.get("TrainingArguments", {
                "dataset_id": dataset_id,
                "model_id": model_id,
                "num_train_epochs": 5,
                "batch_size": 1,
                "per_train_dataset": 0.8,
                "per_test_dataset": 0.2
            })
            cuda_debug = kwargs.get("cuda_debug", False)

            json_file = "training_args.json"
            absolute_path = os.path.abspath(json_file)

            with open(absolute_path, "w") as f:
                json.dump(trainingArguments, f)
                
            logger.info(f"Training arguments: {trainingArguments}")

            if cuda_debug == True:
                os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
                os.environ["NCCL_DEBUG"] = "INFO"

            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
            os.environ["TORCH_USE_CUDA_DSA"] = "0"
            clone_dir = os.path.join(os.getcwd())
            project_id = kwargs.get("project_id", 0)
            token = kwargs.get("token", "hf_JWexoeUOVwf"+"TCxQdNTtLmpGDFIIIUuyeSn")
            checkpoint_version = kwargs.get("checkpoint_version")
            checkpoint_id = kwargs.get("checkpoint")
            dataset_version = kwargs.get("dataset_version")
            dataset = kwargs.get("dataset")
            channel_log = kwargs.get("channel_log", "training_logs")
            world_size = kwargs.get("world_size", 1)
            rank = kwargs.get("rank", 0)
            master_add = kwargs.get("master_add", "127.0.0.1")
            master_port = kwargs.get("master_port", "23456")
            host_name = kwargs.get("host_name", HOST_NAME)
            instruction_field = kwargs.get("prompt_field", "prompt")
            input_field = kwargs.get("input_field", "task_description")
            output_field = kwargs.get("output_field", "response")
            log_queue, logging_thread = start_queue(channel_log)
            write_log(log_queue)
            channel_name = f"{hf_model_id}"
            username = ""
            hf_model_name = ""

            try:
                from huggingface_hub import whoami
                user = whoami(token=push_to_hub_token)['name']
                hf_model_name = f"{user}/{hf_model_id}"
            except Exception as e:
                hf_model_name = "Token not correct"
                print(e)
                
            CHANNEL_STATUS[channel_name] = {
                "status": "training",
                "hf_model_id": hf_model_name,
                "command": command,
                "created_at": time.time(),
            }
            print(f"🚀 Đã bắt đầu training kênh: {channel_name}")

            def func_train_model(
                clone_dir,
                project_id,
                token,
                checkpoint_version,
                checkpoint_id,
                dataset_version,
                dataset,
                model_id,
                world_size,
                rank,
                master_add,
                master_port,
                prompt,
                json_file,
                channel_log,
                hf_model_id,
                push_to_hub,
                push_to_hub_token,
                host_name,
                dataset_id
            ):

                dataset_path = None
                project = connect_project(host_name, token, project_id)

                if dataset_version and dataset and project:
                    dataset_path = os.path.join(
                        clone_dir, f"datasets/{dataset_version}"
                    )

                    if not os.path.exists(dataset_path):
                        data_path = os.path.join(clone_dir, "data_zip")
                        os.makedirs(data_path, exist_ok=True)

                        dataset_name = download_dataset(project, dataset, data_path)
                        print(dataset_name)
                        if dataset_name:
                            data_zip_dir = os.path.join(data_path, dataset_name)

                            with zipfile.ZipFile(data_zip_dir, "r") as zip_ref:
                                zip_ref.extractall(dataset_path)

                            extracted_files = os.listdir(dataset_path)
                            zip_files = [
                                f for f in extracted_files if f.endswith(".zip")
                            ]

                            if len(zip_files) == 1:
                                inner_zip_path = os.path.join(
                                    dataset_path, zip_files[0]
                                )
                                print(
                                    f"🔁 Found inner zip file: {inner_zip_path}, extracting..."
                                )
                                with zipfile.ZipFile(inner_zip_path, "r") as inner_zip:
                                    inner_zip.extractall(dataset_path)
                                os.remove(inner_zip_path)

                subprocess.run(
                    ("whereis accelerate"),
                    shell=True,
                )
                print("===Train===")
                if framework == "huggingface":
                    if int(world_size) > 1:
                        if int(rank) == 0:
                            print("master node")
                            command = (
                                "venv/bin/accelerate launch --num_processes {num_processes} --num_machines {SLURM_NNODES} --machine_rank 0 --main_process_ip {head_node_ip} --main_process_port {port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field} --dataset_id {dataset_id}"
                            ).format(
                                num_processes=world_size * torch.cuda.device_count(),
                                SLURM_NNODES=world_size,
                                head_node_ip=master_add,
                                port=master_port,
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub=push_to_hub,
                                model_id=model_id,
                                push_to_hub_token=push_to_hub_token,
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                                dataset_id=dataset_id
                            )
                            process = subprocess.run(
                                command,
                                shell=True,
                            )
                        else:
                            print("worker node")
                            command = (
                                "venv/bin/accelerate launch --num_processes {num_processes} --num_machines {SLURM_NNODES} --machine_rank {machine_rank} --main_process_ip {head_node_ip} --main_process_port {port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field} --dataset_id {dataset_id}"
                            ).format(
                                num_processes=world_size * torch.cuda.device_count(),
                                SLURM_NNODES=world_size,
                                head_node_ip=master_add,
                                port=master_port,
                                machine_rank=rank,
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub=push_to_hub,
                                model_id=model_id,
                                push_to_hub_token=push_to_hub_token,
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                                dataset_id=dataset_id
                            )
                            process = subprocess.run(
                                command,
                                shell=True,
                            )

                    else:
                        if torch.cuda.device_count() > 1:  # multi gpu
                            command = (
                                "venv/bin/accelerate launch --multi_gpu --num_machines {SLURM_NNODES} --machine_rank 0 --num_processes {num_processes} {file_name} --training_args_json {json_file}  --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field} --dataset_id {dataset_id}"
                            ).format(
                                num_processes=world_size * torch.cuda.device_count(),
                                SLURM_NNODES=world_size,
                                # head_node_ip=os.environ.get("head_node_ip", master_add),
                                port=master_port,
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub=push_to_hub,
                                model_id=model_id,
                                push_to_hub_token=push_to_hub_token,
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                                dataset_id=dataset_id
                            )
                            print("================2")
                            print(command)
                            print("================2")
                            process = subprocess.run(command, shell=True)

                        elif torch.cuda.device_count() == 1:  # one gpu
                            command = (
                                "venv/bin/accelerate launch {file_name} --training_args_json {json_file}  --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field} --dataset_id {dataset_id}"
                            ).format(
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub=push_to_hub,
                                model_id=model_id,
                                push_to_hub_token=push_to_hub_token,
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                                dataset_id=dataset_id
                            )
                            print("================")
                            print(command)
                            print("================")
                            process = subprocess.Popen(
                                command,
                                shell=True,
                                stdout=sys.stdout,
                                stderr=sys.stderr,
                                text=True,
                            )
                            process.wait()
                        else:  # no gpu
                            command = (
                                "venv/bin/accelerate launch --cpu {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field} --dataset_id {dataset_id}"
                            ).format(
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub=push_to_hub,
                                model_id=model_id,
                                push_to_hub_token=push_to_hub_token,
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                                dataset_id=dataset_id
                            )
                            process = subprocess.Popen(
                                command,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                            )
                            while True:
                                output = process.stdout.readline().encode('utf-8')
                                if output == "" and process.poll() is not None:
                                    break
                                if output:
                                    print(output, end="")
                            process.wait()

                elif framework == "pytorch":
                    process = subprocess.run(
                        ("whereis torchrun"),
                        shell=True,
                    )

                    if int(world_size) > 1:
                        if rank == 0:
                            print("master node")
                            command = (
                                "venv/bin/torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                                "--master_addr {master_addr} --master_port {master_port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field} --dataset_id {dataset_id}"
                            ).format(
                                nnodes=int(world_size),
                                node_rank=int(rank),
                                nproc_per_node=world_size * torch.cuda.device_count(),
                                master_addr="127.0.0.1",
                                master_port="23456",
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub=push_to_hub,
                                model_id=model_id,
                                push_to_hub_token=push_to_hub_token,
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                                dataset_id=dataset_id
                            )
                            process = subprocess.Popen(
                                command,
                                shell=True,
                                stdout=sys.stdout,
                                stderr=sys.stderr,
                                text=True,
                            )
                        else:
                            print("worker node")
                            command = (
                                "venv/bin/torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                                "--master_addr {master_addr} --master_port {master_port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field} --dataset_id {dataset_id}"
                            ).format(
                                nnodes=int(world_size),
                                node_rank=int(rank),
                                nproc_per_node=world_size * torch.cuda.device_count(),
                                master_addr=master_add,
                                master_port=master_port,
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub=push_to_hub,
                                model_id=model_id,
                                push_to_hub_token=push_to_hub_token,
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                                dataset_id=dataset_id
                            )
                            print(command)
                            process = subprocess.Popen(
                                command,
                                shell=True,
                                stdout=sys.stdout,
                                stderr=sys.stderr,
                                text=True,
                            )
                    else:
                        command = (
                            "venv/bin/torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                            "{file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field} --dataset_id {dataset_id}"
                        ).format(
                            nnodes=int(world_size),
                            node_rank=int(rank),
                            nproc_per_node=world_size * torch.cuda.device_count(),
                            file_name="./run_distributed_accelerate.py",
                            json_file=json_file,
                            dataset_path=dataset_path,
                            channel_log=channel_log,
                            hf_model_id=hf_model_id,
                            push_to_hub=push_to_hub,
                            model_id=model_id,
                            push_to_hub_token=push_to_hub_token,
                            instruction_field=instruction_field,
                            input_field=input_field,
                            output_field=output_field,
                            dataset_id=dataset_id
                        )
                        process = subprocess.run(
                            command,
                            shell=True,
                        )
                output_dir = "./data/checkpoint"
                print(push_to_hub)
                CHANNEL_STATUS[channel_name]["status"] = "done"
                if push_to_hub:
                    import datetime

                    output_dir = "./data/checkpoint"
                    now = datetime.datetime.now()
                    date_str = now.strftime("%Y%m%d")
                    time_str = now.strftime("%H%M%S")
                    version = f"{date_str}-{time_str}"

                    upload_checkpoint(project, version, output_dir)

            train_thread = threading.Thread(
                target=func_train_model,
                args=(
                    clone_dir,
                    project_id,
                    token,
                    checkpoint_version,
                    checkpoint_id,
                    dataset_version,
                    dataset,
                    model_id,
                    world_size,
                    rank,
                    master_add,
                    master_port,
                    prompt,
                    absolute_path,
                    channel_log,
                    hf_model_id,
                    push_to_hub,
                    push_to_hub_token,
                    host_name,
                    dataset_id
                ),
            )
            train_thread.start()

            return {
                "message": "train completed successfully",
                "channel_name": channel_name,
            }
        elif command.lower() == "stop":
            subprocess.run(["pkill", "-9", "-f", "./inference/generate.py"])
            return {"message": "train stop successfully", "result": "Done"}

        elif command.lower() == "tensorboard":

            def run_tensorboard():
                p = subprocess.Popen(
                    f"tensorboard --logdir /app/data/checkpoint/runs --host 0.0.0.0 --port=6006",
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    text=True,
                )
                out = p.communicate()
                print(out)

            tensorboard_thread = threading.Thread(target=run_tensorboard)
            tensorboard_thread.start()
            return {"message": "tensorboardx started successfully"}

        elif command.lower() == "predict":
            prompt = kwargs.get("prompt", None)
            model_id = kwargs.get("model_id", "google/gemma-3n-e4b-it")
            text = kwargs.get("text", None)
            token_length = kwargs.get("token_lenght", 30)
            task = kwargs.get("task", "")
            voice = kwargs.get("voice", "")
            max_new_token = kwargs.get("max_new_token", 256)
            temperature = kwargs.get("temperature", 0.7)
            top_k = kwargs.get("top_k", 50)
            top_p = kwargs.get("top_p", 0.95)
            raw_input = kwargs.get("input", None)
            docchat_mode = kwargs.get("docchat", False)
            doc_files = kwargs.get("doc_files", None)
            conversation_history = kwargs.get("conversation_history", [])
            session_id = kwargs.get("session_id", None)
            use_history = kwargs.get("use_history", True)
            hf_access_token = kwargs.get(
                "hf_access_token", "hf_JWexoeUOVwfTCxQ"+"dNTtLmpGDFIIIUuyeSn"
            )

            # 🧠 CHAT HISTORY MANAGEMENT
            chat_history = ChatHistoryManager(persist_directory="./chroma_db_history")
            # Auto-create session if not provided
            if not session_id:
                session_result = chat_history.create_new_session()
                session_id = session_result["session_id"]
                print(f"🆕 Created new session: {session_id} with title: {session_result['title']}")
            
            # Load conversation history if enabled
            if use_history and not conversation_history:  # Only load if not already provided
                conversation_history = chat_history.get_session_history(session_id, limit=10)
                if conversation_history:
                    print(f"📚 Loaded {len(conversation_history)} previous conversations for session {session_id}")
                else:
                    print(f"📝 Starting new conversation for session {session_id}")

            # Store original prompt for saving later
            original_prompt = prompt or text

            login(token=hf_access_token)

            if raw_input:
                input_datas = json.loads(raw_input)
                print(input_datas)

            predictions = []

            if not prompt or prompt == "":
                prompt = text

            # Check if any recent conversation history has doc_files
            history_has_docs = False
            history_doc_files = []
            if conversation_history:
                # Duyệt ngược từ lượt nói mới nhất
                for turn in reversed(conversation_history):  
                    turn_doc_files = turn.get('doc_files', [])
                    if turn_doc_files:
                        latest_file = turn_doc_files[-1]  # Lấy file cuối cùng trong lượt nói
                        if latest_file:
                            history_has_docs = True
                            history_doc_files = [latest_file]  # Ghi đè để chỉ giữ file mới nhất
                            print(f"📄 Found latest doc_file in conversation history: {latest_file}")
                            break
            
            def smart_pipeline(
                model_id: str,
                token: str,
                local_dir="./data/checkpoint",
                task="text-generation",
            ):
                global pipe_prediction, processor, model_predict
                model_predict = model_id

                if pipe_prediction == None:
                    try:
                        model_name = model_id.split("/")[-1]
                        local_model_dir = os.path.join(local_dir, model_name)
                        if os.path.exists(local_model_dir) and os.path.exists(
                            os.path.join(local_model_dir, "config.json")
                        ):
                            print(f"✅ Loading model from local: {local_model_dir}")
                            model_source = local_model_dir
                        else:
                            print(f"☁️ Loading model from HuggingFace Hub: {model_id}")
                            model_source = model_id
                    except:
                        print(f"☁️ Loading model from HuggingFace Hub: {model_id}")
                        model_source = model_id

                    processor = AutoProcessor.from_pretrained(model_id)
                    tokenizer = AutoTokenizer.from_pretrained(model_source)
                    dtype = torch.float32
                    if torch.cuda.is_available():
                        if torch.cuda.is_bf16_supported():
                            dtype = torch.bfloat16
                        else:
                            dtype = torch.float16

                        print("Using CUDA.")

                        # load the tokenizer and the model
                        pipe_prediction = Gemma3nForConditionalGeneration.from_pretrained(
                            model_source,
                            device_map="auto",
                            torch_dtype=dtype,
                        ).eval()
                    else:
                        print("Using CPU.")
                        pipe_prediction = Gemma3nForConditionalGeneration.from_pretrained(
                            model_source,
                            device_map="cpu",
                            torch_dtype=dtype,
                        ).eval()

            with torch.no_grad():
                # Load the model
                if not pipe_prediction or model_predict != model_id:
                    smart_pipeline(model_id, hf_access_token)

                # --- DOCCHAT INTEGRATION ---
                if docchat_mode or doc_files or history_has_docs:
                    # doc_files should be a list of file paths
                    if not doc_files:
                        doc_files = []
                    if isinstance(doc_files, str):
                        # If passed as a comma-separated string
                        doc_files = [f.strip() for f in doc_files.split(",") if f.strip()]
                    
                    # If no current doc_files but history has docs, use history docs
                    if not doc_files and history_has_docs:
                        doc_files = history_doc_files
                        print(f"🔄 Using doc_files from conversation history: {doc_files}")
                    
                    # Add conversation history to the prompt if available
                    enhanced_prompt = prompt
                    if conversation_history and not docchat_mode:
                        history_context = chat_history.format_history_for_context(conversation_history, max_turns=3)
                        enhanced_prompt = f"{history_context}\n\nCurrent Question: {prompt}"
                        print(f"🔄 Using conversation history for session {session_id}")
                    
                    answer, verification = "test", "test"
                    
                    print("enhanced_prompt", enhanced_prompt)
                    print("doc_files", doc_files)
                    print("model_id", model_id)

                    answer, verification = docchat_answer(enhanced_prompt, doc_files, model_id, pipe_prediction, tokenizer)
                    if verification != "" or docchat_mode:
                        predictions.append({
                            "result": [
                                {
                                    "from_name": "generated_text",
                                    "to_name": "text_output",
                                    "type": "textarea",
                                    "value": {
                                        "text": [answer],
                                        "thinking": [verification]
                                    },
                                }
                            ],
                            "model_version": "docchat"
                        })
                        
                        # 💾 Save conversation to history (DocChat mode)
                        if use_history and original_prompt and answer:
                            try:
                                # Determine mode based on source of doc_files
                                if history_has_docs and not docchat_mode and not kwargs.get("doc_files"):
                                    mode = "docchat_from_history"
                                else:
                                    mode = "docchat"
                                
                                chat_history.save_conversation_turn(
                                    session_id=session_id,
                                    user_message=original_prompt,
                                    bot_response=answer,
                                    doc_files=doc_files,
                                    metadata={"command": "predict", "mode": mode, "model_id": model_id, "history_docs_used": history_has_docs}
                                )
                                print(f"💾 Saved DocChat conversation to session {session_id} (mode: {mode})")
                            except Exception as e:
                                print(f"❌ Failed to save DocChat conversation: {e}")
                        
                        return {"message": "predict completed successfully (docchat)", "result": predictions, "session_id": session_id}
                # --- END DOCCHAT ---

                # Prepare messages with conversation history
                messages = []
                # Add conversation history if available
                if conversation_history:
                    print(f"🔄 Adding conversation history to messages for session {session_id}")
                    for turn in conversation_history[-3:]:  # Use last 3 turns
                        user_msg = turn.get('user_message', '')
                        bot_response = turn.get('bot_response', '')
                        if user_msg and bot_response:
                            messages.append({"role": "user", "content": [{"type": "text", "text": user_msg}]})
                            messages.append({"role": "assistant", "content": [{"type": "text", "text": bot_response}]})
                
                # Add current user message
                messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
                try:
                    model_inputs = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to(pipe_prediction.device)
                except Exception as e:
                    print(e)

                input_len = model_inputs["input_ids"].shape[-1]

                # conduct text completion
                with torch.inference_mode():
                    generated_ids = pipe_prediction.generate(
                        **model_inputs,
                        max_new_tokens=max_new_token,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        do_sample=False
                    )

                output_ids = generated_ids[0][input_len:]
                thinking_content = ""
                generated_text = processor.decode(output_ids, skip_special_tokens=True).strip("\n")


            print(generated_text)
            predictions.append(
                {
                    "result": [
                        {
                            "from_name": "generated_text",
                            "to_name": "text_output",
                            "type": "textarea",
                            "value": {
                                "thinking": [thinking_content], 
                                "text": [generated_text]
                            },
                        }
                    ],
                    "model_version": "",
                }
            )

            # 💾 Save conversation to history (Normal mode)
            if use_history and original_prompt and generated_text:
                try:
                    chat_history.save_conversation_turn(
                        session_id=session_id,
                        user_message=original_prompt,
                        bot_response=generated_text,
                        doc_files=[],
                        metadata={"command": "predict", "mode": "normal", "model_id": model_id, "thinking": thinking_content}
                    )
                    print(f"💾 Saved normal conversation to session {session_id}")
                except Exception as e:
                    print(f"❌ Failed to save normal conversation: {e}")

            return {"message": "predict completed successfully", "result": predictions, "session_id": session_id}
        
        elif command.lower() == "prompt_sample":
            task = kwargs.get("task", "")
            if task == "question-answering":
                prompt_text = f"""
                    Here is the context: 
                    {{context}}

                    Based on the above context, provide an answer to the following question: 
                    {{question}}

                    Answer:
                    """
            elif task == "text-classification":
                prompt_text = f"""
                    Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

                    Text: 
                    {{context}}

                    Summary:
                    """

            elif task == "summarization":
                prompt_text = f"""
                    Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

                    Text: 
                    {{context}}

                    Summary:
                    """
            return {
                "message": "prompt_sample completed successfully",
                "result": prompt_text,
            }

        elif command.lower() == "action-example":
            return {"message": "Done", "result": "Done"}

        elif command == "status":
            channel = kwargs.get("channel", None)

            if channel:
                # Nếu có truyền kênh cụ thể
                status_info = CHANNEL_STATUS.get(channel)
                if status_info is None:
                    return {"channel": channel, "status": "not_found"}
                elif isinstance(status_info, dict):
                    return {"channel": channel, **status_info}
                else:
                    return {"channel": channel, "status": status_info}
            else:
                # Lấy tất cả kênh
                if not CHANNEL_STATUS:
                    return {"message": "No channels available"}

                channels = []
                for ch, info in CHANNEL_STATUS.items():
                    if isinstance(info, dict):
                        channels.append({"channel": ch, **info})
                    else:
                        channels.append({"channel": ch, "status": info})

                return {"channels": channels}
        else:
            return {"message": "command not supported", "result": None}

    @mcp.tool()
    def model(self, **kwargs):
        global model_demo, processor_demo, model_loaded_demo, model_id_demo

        model_id_demo = kwargs.get("model_id", "google/gemma-3n-e4b-it")
        project_id = kwargs.get("project_id", 0)

        print(
            f"""\
        Project ID: {project_id}
        """
        )
        from huggingface_hub import login

        hf_access_token = kwargs.get(
            "hf_access_token", "hf_JWexoeUOVwfT"+"CxQdNTtLmpGDFIIIUuyeSn"
        )
        login(token=hf_access_token)
        MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

        DESCRIPTION = """\
        # google/gemma-3n-e4b-it
        """

        if not torch.cuda.is_available():
            DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16

        def load_model(model_id, temperature, top_p, top_k, max_new_token):
            print(
                f"""\
                temperature: {temperature}
                top_p: {top_p}
                top_k: {top_k}
                max_new_token: {max_new_token}
                """
            )
            global model_demo, model_loaded_demo, processor_demo

            if torch.cuda.is_available() and not model_loaded_demo:
                model_demo = Gemma3nForConditionalGeneration.from_pretrained(
                    model_id,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=compute_dtype,
                ).eval()

                processor_demo = AutoProcessor.from_pretrained(model_id)
                model_loaded_demo = True
                return f"Model {model_id} loaded successfully!"
            elif model_loaded_demo:
                return "Model is already loaded! Please refresh the page to load a different model."
            else:
                return "Error: CUDA is not available!"
            
        @spaces.GPU
        def generate(
            message: str,
            history,
            system_prompt: str,
            max_new_tokens: int = 1024,
            temperature: float = 0.6,
            top_p: float = 0.9,
            top_k: int = 50,
            repetition_penalty: float = 1,
        ) -> Iterator[str]:
            if not model_loaded_demo:
                return (
                    "Please load the model first by clicking the 'Load Model' button."
                )
            chat_messages = []
            input_message = None
            if system_prompt:
                chat_messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

            for user_turn, assistant_turn in history:
                user_content = []
                if isinstance(user_turn, tuple):
                    image_path = user_turn[0]
                    user_content.append({"type": "image", "image": image_path})
                elif isinstance(user_turn, str):
                    user_content.append({"type": "text", "text": user_turn})

                if user_content:
                    chat_messages.append({"role": "user", "content": user_content})
                    chat_messages.append({"role": "assistant", "content": [{"type": "text", "text": str(assistant_turn)}]})

            current_user_content = []
            if isinstance(message, list):
                input_message = message[0]
            else:
                input_message = message

            if input_message.get('files'):
                image_path = input_message['files'][0]
                current_user_content.append({"type": "image", "image": image_path})
            if input_message.get('text'):
                current_user_content.append({"type": "text", "text": input_message['text']})

            chat_messages.append({"role": "user", "content": current_user_content})
            print(f"Chat messages: {chat_messages}")

            model_inputs = processor_demo.apply_chat_template(
                chat_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model_demo.device)

            input_len = model_inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generated_ids = model_demo.generate(
                    **model_inputs, 
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=False
                )

            output_ids = generated_ids[0][input_len:]
            response = processor_demo.decode(output_ids, skip_special_tokens=True)
            return response

        BEE_IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
        
        examples = [
            [{"text": "Describe this image", "files": [BEE_IMAGE_URL]}],
            [{"text": "Can you explain briefly to me what is the Python programming language?"}],
            [{"text": "write a program to find the factorial of a number"}],
        ]

        multimodal_textbox = gr.MultimodalTextbox(
            file_types=["image"],
            placeholder="Input text or image...",
            show_label=False,
        )

        with gr.Blocks(css="style.css") as demo:
            gr.Markdown(DESCRIPTION)
            with gr.Row():
                with gr.Column(scale=1):
                    load_btn = gr.Button("Load Model")
                with gr.Column(scale=1):
                    status_text = gr.Textbox(label="Model Status", interactive=False)

            with gr.Accordion("Advanced Options", open=False):
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    placeholder="Enter system prompt here...",
                    lines=2,
                    value="You are a helpful assistant.",
                )
                temperature = gr.Slider(
                    label="Temperature", minimum=0.0, maximum=100.0, step=0.1, value=0.9
                )
                top_p = gr.Slider(
                    label="Top_p", minimum=0.0, maximum=1.0, step=0.1, value=0.6
                )
                top_k = gr.Slider(
                    label="Top_k", minimum=0, maximum=100, step=1, value=0
                )
                max_new_token = gr.Slider(
                    label="Max new tokens", minimum=1, maximum=1024, step=1, value=256
                )
            load_btn.click(fn=lambda: load_model(model_id_demo, temperature.value, top_p, top_k, max_new_token), outputs=status_text)
    
            gr.ChatInterface(
                fn=generate,
                additional_inputs=[system_prompt, max_new_token, temperature, top_p, top_k, ],
                chatbot=gr.Chatbot(
                    label="Gemma 3N Chat",
                    show_label=False,
                    container=False,
                    show_copy_button=True,
                    bubble_full_width=False,
                    layout="bubble",
                    height=500,
                ),
                textbox=multimodal_textbox,
                stop_btn=gr.Button("Dừng lại"),
                examples=examples,
                multimodal=True
            )

        gradio_app, local_url, share_url = demo.launch(
            share=True,
            quiet=True,
            prevent_thread_lock=True,
            server_name="0.0.0.0",
            show_error=True,
        )

        return {"share_url": share_url, "local_url": local_url}

    @mcp.tool()
    def model_trial(self, project, **kwargs):
        return {"message": "Done", "result": "Done"}

    @mcp.tool()
    def download(self, project, **kwargs):
        from flask import request, send_from_directory

        file_path = request.args.get("path")
        print(request.args)
        return send_from_directory(os.getcwd(), file_path)
    
    @mcp.tool()
    def model_docchat(self, **kwargs):

        css = """
        .title { font-size: 1.5em !important; text-align: center !important; color: #FFD700; }
        .subtitle { font-size: 1em !important; text-align: center !important; color: #FFD700; }
        .text { text-align: center; }
        """

        with gr.Blocks(css=css, title="Gemma3 DocChat 🐥") as demo:
            gr.Markdown("## Gemma3 DocChat: Document Q&A with Fact Verification", elem_classes="subtitle")
            gr.Markdown("# How it works ✨:", elem_classes="title")
            gr.Markdown("📤 Upload your document(s), enter your query then hit Submit 📝", elem_classes="text")
            gr.Markdown("⚠️ **Note:** Only accepts: .pdf, .docx, .txt, .md", elem_classes="text")

            with gr.Row():
                with gr.Column():
                    files = gr.Files(label="📄 Upload Documents", file_types=constants.ALLOWED_TYPES)
                    question = gr.Textbox(label="❓ Question", lines=3)
                    submit_btn = gr.Button("Submit 🚀")
                with gr.Column():
                    answer_output = gr.Textbox(label="🐥 Answer", interactive=False)
                    verification_output = gr.Textbox(label="✅ Verification Report", interactive=False)

            def process_docchat(question_text, uploaded_files):
                if not question_text or not question_text.strip():
                    return "❌ Question cannot be empty", ""
                if not uploaded_files:
                    return "❌ No documents uploaded", ""
                file_paths = [f.name for f in uploaded_files if hasattr(f, 'name') and os.path.exists(f.name)]
                answer, verification = docchat_answer(question_text, file_paths)
                return answer, verification

            submit_btn.click(
                fn=process_docchat,
                inputs=[question, files],
                outputs=[answer_output, verification_output]
            )

        gradio_app, local_url, share_url = demo.launch(
            share=True,
            quiet=True,
            prevent_thread_lock=True,
            server_name="0.0.0.0",
            show_error=True,
        )
        return {"share_url": share_url, "local_url": local_url}
