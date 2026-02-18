import gymnasium as gym
import random
import numpy as np
import os
import time
import subprocess
from jinja2 import Template
# from openai import OpenAI
# import google.generativeai as genai
# import anthropic
import ollama
import requests


class LLMBrain:
    def __init__(
        self,
        llm_si_template: Template,
        llm_output_conversion_template: Template,
        llm_model_name: str,
        ollama_num_ctx: int = 4096,
        ollama_num_predict: int = None,
    ):
        self.llm_si_template = llm_si_template
        self.llm_output_conversion_template = llm_output_conversion_template
        self.llm_conversation = []
        # Support both cloud models and local Ollama models
        CLOUD_MODELS = [
            "o1-preview",
            "gpt-4o",
            "gemini-2.0-flash-exp",
            "gpt-4o-mini",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
            "gemini-2.5-pro-preview-05-06",
            "gemini-2.5-flash-preview-04-17",
            "o3-mini-2025-01-31",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "claude-3-7-sonnet-20250219",
        ]
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LMSTUDIO_BASE_URL")
        if (
            llm_model_name not in CLOUD_MODELS
            and "ollama" not in llm_model_name.lower()
            and not base_url
        ):
            raise ValueError(
                f"Unknown model: {llm_model_name}. Use a cloud model, set OPENAI_BASE_URL, or prefix with 'ollama:'"
            )
        
        is_ollama_prefixed = llm_model_name.lower().startswith("ollama:")
        normalized_model_name = llm_model_name.split(":", 1)[1] if is_ollama_prefixed else llm_model_name
        self.llm_model_name = llm_model_name
        self.ollama_num_ctx = ollama_num_ctx
        self.ollama_num_predict = ollama_num_predict
        if "gemini" in llm_model_name:
            self.model_group = "gemini"
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        elif "claude" in llm_model_name:
            self.model_group = "anthropic"
            self.client = anthropic.Client(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif "gpt-oss" in normalized_model_name.lower() and base_url:
            self.model_group = "openai"
            self.llm_model_name = normalized_model_name
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "OpenAI client not installed. Install openai to use OPENAI_BASE_URL with gpt-oss."
                ) from exc
            self.client = OpenAI(
                base_url=base_url,
                api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
            )
        elif llm_model_name.lower().startswith("ollama:"):
            self.model_group = "ollama"
            # Extract model name after 'ollama:' prefix
            self.ollama_model = llm_model_name.split(":", 1)[1]
        else:
            self.model_group = "openai"
            self.client = OpenAI()

    def reset_llm_conversation(self):
        self.llm_conversation = []

    def _count_tokens_ollama(self, text: str) -> int:
        """Use Ollama's tokenize API to get exact token count for the current model.
        Falls back to heuristic if API not available."""
        try:
            url = "http://localhost:11434/api/tokenize"
            payload = {"model": self.ollama_model, "content": text}
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            tokens = response.json().get("tokens", [])
            token_count = len(tokens)
            print(f"[TOKENS] {self.ollama_model}: {token_count} tokens")
            return token_count
        except Exception:
            # Fallback: estimate using word/token heuristic (~1.3 tokens per word)
            return max(1, int(len(text.split()) * 1.3))

    def _ollama_prompt_guard(self, prompt_text: str, episode_reward_buffer, step_number, num_episodes):
        """Context guard disabled - no longer checking for exceeding context window."""
        # Guard check disabled per user request
        return

    def add_llm_conversation(self, text, role):
        if self.model_group in ["openai", "ollama"]:
            self.llm_conversation.append({"role": role, "content": text})
        elif self.model_group == "anthropic":
            self.llm_conversation.append({"role": role, "content": text})
        else:
            self.llm_conversation.append({"role": role, "parts": text})

    def query_llm(self):
        for attempt in range(10):
            try:
                if self.model_group == "openai":
                    completion = self.client.chat.completions.create(
                        model=self.llm_model_name,
                        messages=self.llm_conversation,
                    )
                    message = completion.choices[0].message
                    response = message.content
                    # LM Studio / gpt-oss may return reasoning separately
                    reasoning = getattr(message, "thinking", None)
                    if reasoning:
                        response = f"<think>{reasoning}</think>\n{response}"
                elif self.model_group == "ollama":
                    # Stream tokens in real-time from Ollama
                    content = ""
                    full_thinking = ""
                    thinking_started = False
                    ollama_options = {"num_ctx": self.ollama_num_ctx, "num_gpu": -1}
                    if self.ollama_num_predict is not None:
                        ollama_options["num_predict"] = self.ollama_num_predict
                    
                    print("\n[OLLAMA STREAMING] ", end="", flush=True)
                    
                    # Use stream=True to get tokens as they're generated
                    stream = ollama.chat(
                        model=self.ollama_model,
                        messages=self.llm_conversation,
                        options=ollama_options,
                        stream=True
                    )
                    
                    for chunk in stream:
                        message = chunk.get("message", {})
                        
                        # Handle thinking tokens (if present)
                        if "thinking" in message:
                            thinking_token = message.get("thinking", "")
                            if thinking_token:
                                if not thinking_started:
                                    print("<think>", end="", flush=True)
                                    thinking_started = True
                                full_thinking += thinking_token
                                print(thinking_token, end="", flush=True)
                        
                        # Handle regular content tokens
                        if "content" in message:
                            token = message.get("content", "")
                            if token:
                                if thinking_started:
                                    print("</think>", end="", flush=True)
                                    thinking_started = False
                                content += token
                                print(token, end="", flush=True)
                    
                    # Close thinking tag if it's still open
                    if thinking_started:
                        print("</think>", end="", flush=True)
                    
                    print("\n")  # New line after streaming completes
                    
                    if full_thinking:
                        response = f"<think>{full_thinking}</think>\n{content}"
                    else:
                        response = content
                elif self.model_group == "anthropic":
                    message = self.client.messages.create(
                        model=self.llm_model_name,
                        messages=self.llm_conversation,
                        max_tokens=1024,
                    )
                    response = message.content[0].text
                else:
                    model = genai.GenerativeModel(model_name=self.llm_model_name)
                    chat_session = model.start_chat(history=self.llm_conversation[:-1])
                    response = chat_session.send_message(
                        self.llm_conversation[-1]["parts"]
                    )
                    response = response.text
            except Exception as e:
                print(f"Error: {e}")
                print("Retrying...")
                if attempt == 9:
                    raise Exception("Failed")
                else:
                    print("Waiting for 60 seconds before retrying...")
                    time.sleep(60)

            if self.model_group in ["openai", "ollama"]:
                # add the response to self.llm_conversation
                self.add_llm_conversation(response, "assistant")
            else:
                self.add_llm_conversation(response, "model")

            return response

    def query_llm_multiple_response(self, num_responses, temperature):
        for attempt in range(10):
            try:
                if self.model_group == "openai":
                    completion = self.client.chat.completions.create(
                        model=self.llm_model_name,
                        messages=self.llm_conversation,
                        n=num_responses,
                        temperature=temperature,
                    )
                    responses = []
                    for i in range(num_responses):
                        message = completion.choices[i].message
                        content = message.content
                        reasoning = getattr(message, "reasoning", None)
                        if reasoning:
                            content = f"<think>{reasoning}</think>\n{content}"
                        responses.append(content)
                elif self.model_group == "ollama":
                    # Ollama doesn't support multiple responses natively,
                    # so we generate them sequentially with streaming
                    responses = []
                    ollama_options = {"num_ctx": self.ollama_num_ctx, "num_gpu": -1}
                    if self.ollama_num_predict is not None:
                        ollama_options["num_predict"] = self.ollama_num_predict
                    for resp_idx in range(num_responses):
                        content = ""
                        full_thinking = ""
                        thinking_started = False
                        
                        print(f"\n[OLLAMA RESPONSE {resp_idx + 1}/{num_responses}] ", end="", flush=True)
                        
                        # Stream this response
                        stream = ollama.chat(
                            model=self.ollama_model,
                            messages=self.llm_conversation,
                            options=ollama_options,
                            stream=True
                        )
                        
                        for chunk in stream:
                            message = chunk.get("message", {})
                            
                            # Handle thinking tokens (print in real-time)
                            if "thinking" in message:
                                thinking_token = message.get("thinking", "")
                                if thinking_token:
                                    if not thinking_started:
                                        print("<think>", end="", flush=True)
                                        thinking_started = True
                                    full_thinking += thinking_token
                                    print(thinking_token, end="", flush=True)
                            
                            # Handle regular content tokens
                            if "content" in message:
                                token = message.get("content", "")
                                if token:
                                    if thinking_started:
                                        print("</think>", end="", flush=True)
                                        thinking_started = False
                                    content += token
                                    print(token, end="", flush=True)
                        
                        # Close thinking tag if it's still open
                        if thinking_started:
                            print("</think>", end="", flush=True)
                        
                        print("\n")  # New line after each response
                        
                        if full_thinking:
                            responses.append(f"<think>{full_thinking}</think>\n{content}")
                        else:
                            responses.append(content)
                else:
                    model = genai.GenerativeModel(model_name=self.llm_model_name)
                    responses = model.generate_content(
                        contents=self.llm_conversation,
                        generation_config=genai.GenerationConfig(
                            candidate_count=num_responses,
                            temperature=temperature,
                        ),
                    )
                    responses = [
                        "\n".join([x.text for x in c.content.parts])
                        for c in responses.candidates
                    ]

            except Exception as e:
                print(f"Error: {e}")
                print("Retrying...")
                if attempt == 4:
                    raise Exception("Failed")
                else:
                    print("Waiting for 60 seconds before retrying...")
                    time.sleep(60)

            return responses

    def parse_parameters(self, parameters_string):
        new_parameters_list = []

        # Update the Q-table based on the new Q-table
        for row in parameters_string.split("\n"):
            if row.strip().strip(","):
                try:
                    parameters_row = [
                        float(x.strip().strip(",")) for x in row.split(",")
                    ]
                    new_parameters_list.append(parameters_row)
                except Exception as e:
                    print(e)

        return new_parameters_list

    def llm_update_parameters(self, parameters, replay_buffer, parse_parameters=None):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "replay_buffer_string": str(replay_buffer),
                "parameters_string": str(parameters),
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        if self.model_group in ["openai", "ollama"]:
            self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        else:
            self.add_llm_conversation(new_parameters_with_reasoning, "model")
        self.add_llm_conversation(
            self.llm_output_conversion_template.render(),
            "user",
        )
        new_parameters = self.query_llm()

        if parse_parameters is None:
            new_parameters_list = self.parse_parameters(new_parameters)
        else:
            new_parameters_list = parse_parameters(new_parameters)

        return new_parameters_list, [new_parameters_with_reasoning, new_parameters]

    def llm_update_parameters_sas(self, episode_reward_buffer, parse_parameters=None):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {"episode_reward_buffer_string": str(episode_reward_buffer)}
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        print(system_prompt)

        self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        self.add_llm_conversation(
            self.llm_output_conversion_template.render(),
            "user",
        )
        new_parameters = self.query_llm()

        if parse_parameters is None:
            new_parameters_list = self.parse_parameters(new_parameters)
        else:
            new_parameters_list = parse_parameters(new_parameters)

        return new_parameters_list, [
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
            new_parameters,
        ]

    def llm_update_parameters_num_optim(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        rank=None,
        optimum=None,
        search_step_size=0.1,
        actions=None,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "rank": rank,
                "optimum": str(optimum),
                "step_size": str(search_step_size),
                "actions": actions,
            }
        )

        self.add_llm_conversation(system_prompt, "user")

        api_start_time = time.time()
        new_parameters_with_reasoning = self.query_llm()
        api_time = time.time() - api_start_time

        # print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
            api_time,
        )

    def llm_update_parameters_num_optim_q_table(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        actions,
        num_states,
        optimum,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "actions": actions,
                "rank": num_states,
                "optimum": str(optimum),
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
        )

    def llm_update_parameters_num_optim_imitation(
        self,
        demonstrations_str,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        search_std,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "expert_demonstration_string": demonstrations_str,
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "search_std": str(search_std),
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
        )

    def llm_propose_parameters_num_optim_based_on_anchor(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        search_std,
        anchor_parameters,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "search_std": str(search_std),
                "anchor_parameters": str(anchor_parameters),
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
        )

    def llm_propose_multiple_parameters_num_optim_based_on_anchor(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        search_std,
        anchor_parameters,
        num_candidates,
        temperature,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "search_std": str(search_std),
                "anchor_parameters": str(anchor_parameters),
            }
        )

        # print(system_prompt)
        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning_list = self.query_llm_multiple_response(
            num_candidates, temperature
        )
        # print(new_parameters_with_reasoning_list)

        new_parameters_list = []
        reasonings_list = []
        for new_params in new_parameters_with_reasoning_list:
            new_params_np = parse_parameters(new_params)
            new_parameters_list.append(new_params_np)
            reasonings_list.append(new_params)

        return (
            system_prompt,
            new_parameters_list,
            reasonings_list,
        )

    def llm_propose_parameters_num_optim_based_on_anchor_thread(
        self,
        new_candidates,
        new_idx,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        search_std,
        anchor_parameters,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "search_std": str(search_std),
                "anchor_parameters": str(anchor_parameters),
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)
        new_candidates[new_idx] = new_parameters_list

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
        )

    def llm_update_parameters_num_optim_semantics(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        env_desc_file,
        num_episodes=400,
        rank=None,
        optimum=None,
        search_step_size=0.1,
        actions=None,
        num_evaluation_episodes=20,
        attempt_idx=0,
    ):
        self.reset_llm_conversation()

        full_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "env_description": env_desc_file,
                "step_number": str(step_number),
                "num_episodes": num_episodes,
                "rank": rank,
                "optimum": str(optimum),
                "step_size": str(search_step_size),
                "actions": actions,
                "num_evaluation_episodes": num_evaluation_episodes,
                "attempt_idx": attempt_idx,
            }
        )
        
        # Track context size
        context_size = 0
        
        # For Ollama: Split into system (static rules) and user (dynamic data) messages
        # For other models: Keep current behavior (single user message)
        if self.model_group == "ollama":
            split_marker = "Next, you will see examples of params and their episodic reward f(params)."
            parts = full_prompt.split(split_marker)
            
            if len(parts) == 2:
                system_part = parts[0].strip()
                user_part = split_marker + "\n" + parts[1].strip()
                
                # Add system message first (high-priority instructions)
                self.add_llm_conversation(system_part, "system")
                
                # Count tokens for logging (guard check disabled)
                combined_prompt = system_part + "\n\n" + user_part
                context_size = self._count_tokens_ollama(combined_prompt)
                
                # Add user message (dynamic data: examples, iteration, warnings)
                self.add_llm_conversation(user_part, "user")
                
                # For logging, reconstruct full prompt
                system_prompt = "SYSTEM:\n" + system_part + "\n\nUSER:\n" + user_part
            else:
                # Fallback if split fails
                context_size = self._count_tokens_ollama(full_prompt)
                self.add_llm_conversation(full_prompt, "user")
                system_prompt = full_prompt
        else:
            # Cloud APIs: use current behavior (single user message)
            # Estimate context size for non-Ollama models (rough approximation)
            context_size = max(1, int(len(full_prompt.split()) * 1.3))
            self.add_llm_conversation(full_prompt, "user")
            system_prompt = full_prompt

        api_start_time = time.time()
        new_parameters_with_reasoning = self.query_llm()
        api_time = time.time() - api_start_time

        # print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            system_prompt + "\n\n\nLLM:\n" + new_parameters_with_reasoning,
            api_time,
            context_size,
        )

    def llm_update_parameters_with_feedback(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        env_desc_file,
        num_episodes=400,
        rank=None,
        optimum=None,
        search_step_size=0.1,
        actions=None,
        num_evaluation_episodes=20,
        attempt_idx=0,
        human_feedback="",
        last_policy_params="",
        request_feedback_prediction=False,
    ):
        """
        Update policy parameters with human feedback from the loop.
        
        This method is similar to llm_update_parameters_num_optim_semantics but
        includes human feedback and last policy parameters in the prompt.
        """
        self.reset_llm_conversation()

        full_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "env_description": env_desc_file,
                "step_number": str(step_number),
                "num_episodes": num_episodes,
                "rank": rank,
                "optimum": str(optimum),
                "step_size": str(search_step_size),
                "actions": actions,
                "num_evaluation_episodes": num_evaluation_episodes,
                "attempt_idx": attempt_idx,
                "human_feedback": human_feedback,
                "last_policy_params": last_policy_params,
                "request_feedback_prediction": request_feedback_prediction,
            }
        )
        
        # Track context size
        context_size = 0
        
        # For Ollama: Split into system (static rules) and user (dynamic data) messages
        # For other models: Keep current behavior (single user message)
        if self.model_group == "ollama":
            split_marker = "Next, you will see examples of params"
            parts = full_prompt.split(split_marker)
            
            if len(parts) == 2:
                system_part = parts[0].strip()
                user_part = split_marker + parts[1].strip()
                
                # Add system message first (high-priority instructions)
                self.add_llm_conversation(system_part, "system")
                
                # Count tokens for logging (guard check disabled)
                combined_prompt = system_part + "\n\n" + user_part
                context_size = self._count_tokens_ollama(combined_prompt)
                
                # Add user message (dynamic data: examples, iteration, warnings, feedback)
                self.add_llm_conversation(user_part, "user")
                
                # For logging, reconstruct full prompt
                system_prompt = "SYSTEM:\n" + system_part + "\n\nUSER:\n" + user_part
            else:
                # Fallback if split fails
                context_size = self._count_tokens_ollama(full_prompt)
                self.add_llm_conversation(full_prompt, "user")
                system_prompt = full_prompt
        else:
            # Cloud APIs: use current behavior (single user message)
            # Estimate context size for non-Ollama models (rough approximation)
            context_size = max(1, int(len(full_prompt.split()) * 1.3))
            self.add_llm_conversation(full_prompt, "user")
            system_prompt = full_prompt

        api_start_time = time.time()
        new_parameters_with_reasoning = self.query_llm()
        api_time = time.time() - api_start_time

        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            system_prompt + "\n\n\nLLM:\n" + new_parameters_with_reasoning,
            api_time,
            context_size,
        )
