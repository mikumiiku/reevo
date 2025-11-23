from typing import Optional
import logging
import subprocess
import numpy as np
import os
from omegaconf import DictConfig

from utils.utils import *
from utils.llm_client.base import BaseClient
from utils.lineage_tracker import LineageTracker


class ReEvo:
    def __init__(
        self, 
        cfg: DictConfig, 
        root_dir: str, 
        generator_llm: BaseClient, 
        reflector_llm: Optional[BaseClient] = None,
        
        # Support setting different LLMs for each of the four operators: 
        # Short-term Reflection, Long-term Reflection, Crossover, Mutation
        short_reflector_llm: Optional[BaseClient] = None,
        long_reflector_llm: Optional[BaseClient] = None,
        crossover_llm: Optional[BaseClient] = None,
        mutation_llm: Optional[BaseClient] = None
    ) -> None:
        self.cfg = cfg
        self.generator_llm = generator_llm
        self.reflector_llm = reflector_llm or generator_llm

        self.short_reflector_llm = short_reflector_llm or self.reflector_llm
        self.long_reflector_llm = long_reflector_llm or self.reflector_llm
        self.crossover_llm = crossover_llm or generator_llm
        self.mutation_llm = mutation_llm or generator_llm

        self.root_dir = root_dir
        
        self.mutation_rate = cfg.mutation_rate
        self.iteration = 0
        self.function_evals = 0
        self.elitist = None
        self.long_term_reflection_str = ""
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None
        
        # 初始化谱系追踪器
        self.lineage_tracker = LineageTracker()
        
        # 谱系选择配置（从config读取，提供默认值）
        self.enable_lineage_selection = cfg.get('enable_lineage_selection', True)
        self.elite_ratio = cfg.get('elite_ratio', 0.5)
        self.lineage_ratio = cfg.get('lineage_ratio', 0.3)
        self.revival_ratio = cfg.get('revival_ratio', 0.2)
        self.clade_score_alpha = cfg.get('clade_score_alpha', 0.5)
        self.clade_score_beta = cfg.get('clade_score_beta', 0.3)
        self.clade_score_gamma = cfg.get('clade_score_gamma', 0.2)
        
        logging.info(f"Lineage selection enabled: {self.enable_lineage_selection}")
        if self.enable_lineage_selection:
            logging.info(f"Selection ratios - Elite: {self.elite_ratio}, Lineage: {self.lineage_ratio}, Revival: {self.revival_ratio}")
        
        self.init_prompt()
        self.init_population()


    def init_prompt(self) -> None:
        self.problem = self.cfg.problem.problem_name
        self.problem_desc = self.cfg.problem.description
        self.problem_size = self.cfg.problem.problem_size
        self.func_name = self.cfg.problem.func_name
        self.obj_type = self.cfg.problem.obj_type
        self.problem_type = self.cfg.problem.problem_type
        
        logging.info("Problem: " + self.problem)
        logging.info("Problem description: " + self.problem_desc)
        logging.info("Function name: " + self.func_name)
        
        self.prompt_dir = f"{self.root_dir}/prompts"
        self.output_file = f"{self.root_dir}/problems/{self.problem}/gpt.py"
        
        # Loading all text prompts
        # Problem-specific prompt components
        prompt_path_suffix = "_black_box" if self.problem_type == "black_box" else ""
        problem_prompt_path = f'{self.prompt_dir}/{self.problem}{prompt_path_suffix}'
        self.seed_func = file_to_string(f'{problem_prompt_path}/seed_func.txt')
        self.func_signature = file_to_string(f'{problem_prompt_path}/func_signature.txt')
        self.func_desc = file_to_string(f'{problem_prompt_path}/func_desc.txt')
        if os.path.exists(f'{problem_prompt_path}/external_knowledge.txt'):
            self.external_knowledge = file_to_string(f'{problem_prompt_path}/external_knowledge.txt')
            self.long_term_reflection_str = self.external_knowledge
        else:
            self.external_knowledge = ""
        
        
        # Common prompts
        self.system_generator_prompt = file_to_string(f'{self.prompt_dir}/common/system_generator.txt')
        self.system_reflector_prompt = file_to_string(f'{self.prompt_dir}/common/system_reflector.txt')
        self.user_reflector_st_prompt = file_to_string(f'{self.prompt_dir}/common/user_reflector_st.txt') if self.problem_type != "black_box" else file_to_string(f'{self.prompt_dir}/common/user_reflector_st_black_box.txt') # shrot-term reflection
        self.user_reflector_lt_prompt = file_to_string(f'{self.prompt_dir}/common/user_reflector_lt.txt') # long-term reflection
        self.crossover_prompt = file_to_string(f'{self.prompt_dir}/common/crossover.txt')
        self.mutation_prompt = file_to_string(f'{self.prompt_dir}/common/mutation.txt')
        self.user_generator_prompt = file_to_string(f'{self.prompt_dir}/common/user_generator.txt').format(
            func_name=self.func_name, 
            problem_desc=self.problem_desc,
            func_desc=self.func_desc,
            )
        self.seed_prompt = file_to_string(f'{self.prompt_dir}/common/seed.txt').format(
            seed_func=self.seed_func,
            func_name=self.func_name,
        )

        # Flag to print prompts
        self.print_crossover_prompt = True # Print crossover prompt for the first iteration
        self.print_mutate_prompt = True # Print mutate prompt for the first iteration
        self.print_short_term_reflection_prompt = True # Print short-term reflection prompt for the first iteration
        self.print_long_term_reflection_prompt = True # Print long-term reflection prompt for the first iteration


    def init_population(self) -> None:
        # Evaluate the seed function, and set it as Elite
        logging.info("Evaluating seed function...")
        code = extract_code_from_generator(self.seed_func).replace("v1", "v2")
        logging.info("Seed function code: \n" + code)
        seed_ind = {
            "stdout_filepath": f"problem_iter{self.iteration}_stdout0.txt",
            "code_path": f"problem_iter{self.iteration}_code0.py",
            "code": code,
            "response_id": 0,
            "birth_iteration": self.iteration,
        }
        self.seed_ind = seed_ind
        self.population = self.evaluate_population([seed_ind])

        # If seed function is invalid, stop
        if not self.seed_ind["exec_success"]:
            raise RuntimeError(f"Seed function is invalid. Please check the stdout file in {os.getcwd()}.")

        # 注册种子个体到谱系追踪器
        self.lineage_tracker.register_individual(self.seed_ind, parents=None)
        logging.info(f"Registered seed individual to lineage tracker")

        self.update_iter()
        
        # Generate responses
        system = self.system_generator_prompt
        user = self.user_generator_prompt + "\n" + self.seed_prompt + "\n" + self.long_term_reflection_str
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        logging.info("Initial Population Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)

        responses = self.generator_llm.multi_chat_completion([messages], self.cfg.init_pop_size, temperature = self.generator_llm.temperature + 0.3) # Increase the temperature for diverse initial population
        population = [self.response_to_individual(response, response_id) for response_id, response in enumerate(responses)]

        # Run code and evaluate population
        population = self.evaluate_population(population)

        # 注册初始种群到谱系追踪器（每个建立独立谱系）
        for ind in population:
            self.lineage_tracker.register_individual(ind, parents=None)
        logging.info(f"Registered {len(population)} initial individuals to lineage tracker")

        # Update iteration
        self.population = population
        self.update_iter()

    
    def response_to_individual(self, response: str, response_id: int, file_name: str=None) -> dict:
        """
        Convert response to individual
        """
        # Write response to file
        file_name = f"problem_iter{self.iteration}_response{response_id}.txt" if file_name is None else file_name + ".txt"
        with open(file_name, 'w') as file:
            file.writelines(response + '\n')

        code = extract_code_from_generator(response)

        # Extract code and description from response
        std_out_filepath = f"problem_iter{self.iteration}_stdout{response_id}.txt" if file_name is None else file_name + "_stdout.txt"
        
        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": f"problem_iter{self.iteration}_code{response_id}.py",
            "code": code,
            "response_id": response_id,
            "birth_iteration": self.iteration,
        }
        return individual

    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """
        Mark an individual as invalid.
        """
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["traceback_msg"] = traceback_msg
        return individual


    def evaluate_population(self, population: list[dict]) -> list[float]:
        """
        Evaluate population by running code in parallel and computing objective values.
        """
        inner_runs = []
        # Run code to evaluate
        for response_id in range(len(population)):
            self.function_evals += 1
            # Skip if response is invalid
            if population[response_id]["code"] is None:
                population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid response!")
                inner_runs.append(None)
                continue
            
            logging.info(f"Iteration {self.iteration}: Running Code {response_id}")
            
            try:
                process = self._run_code(population[response_id], response_id)
                inner_runs.append(process)
            except Exception as e: # If code execution fails
                logging.info(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                inner_runs.append(None)
        
        # Update population with objective values
        for response_id, inner_run in enumerate(inner_runs):
            if inner_run is None: # If code execution fails, skip
                continue
            try:
                inner_run.communicate(timeout=self.cfg.timeout) # Wait for code execution to finish
            except subprocess.TimeoutExpired as e:
                logging.info(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                inner_run.kill()
                continue

            individual = population[response_id]
            stdout_filepath = individual["stdout_filepath"]
            with open(stdout_filepath, 'r') as f:  # read the stdout file
                stdout_str = f.read() 
            traceback_msg = filter_traceback(stdout_str)
            
            individual = population[response_id]
            # Store objective value for each individual
            if traceback_msg == '': # If execution has no error
                try:
                    individual["obj"] = float(stdout_str.split('\n')[-2]) if self.obj_type == "min" else -float(stdout_str.split('\n')[-2])
                    individual["exec_success"] = True
                except:
                    population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid std out / objective value!")
            else: # Otherwise, also provide execution traceback error feedback
                population[response_id] = self.mark_invalid_individual(population[response_id], traceback_msg)

            logging.info(f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual['obj']}")
        return population


    def _run_code(self, individual: dict, response_id) -> subprocess.Popen:
        """
        Write code into a file and run eval script.
        """
        logging.debug(f"Iteration {self.iteration}: Processing Code Run {response_id}")
        
        with open(self.output_file, 'w') as file:
            file.writelines(individual["code"] + '\n')

        # Execute the python file with flags
        with open(individual["stdout_filepath"], 'w') as f:
            eval_file_path = f'{self.root_dir}/problems/{self.problem}/eval.py' if self.problem_type != "black_box" else f'{self.root_dir}/problems/{self.problem}/eval_black_box.py' 
            process = subprocess.Popen(['python', '-u', eval_file_path, f'{self.problem_size}', self.root_dir, "train"],
                                        stdout=f, stderr=f)

        block_until_running(individual["stdout_filepath"], log_status=True, iter_num=self.iteration, response_id=response_id)
        return process

    
    def update_iter(self) -> None:
        """
        Update after each iteration
        """
        population = self.population
        objs = [individual["obj"] for individual in population]
        best_obj, best_sample_idx = min(objs), np.argmin(np.array(objs))
        
        # update best overall
        if self.best_obj_overall is None or best_obj < self.best_obj_overall:
            self.best_obj_overall = best_obj
            self.best_code_overall = population[best_sample_idx]["code"]
            self.best_code_path_overall = population[best_sample_idx]["code_path"]
        
        # update elitist
        if self.elitist is None or best_obj < self.elitist["obj"]:
            self.elitist = population[best_sample_idx]
            logging.info(f"Iteration {self.iteration}: Elitist: {self.elitist['obj']}")
        
        best_path = self.best_code_path_overall.replace(".py", ".txt").replace("code", "response")
        logging.info(f"Best obj: {self.best_obj_overall}, Best Code Path: {print_hyperlink(best_path, self.best_code_path_overall)}")
        logging.info(f"Iteration {self.iteration} finished...")
        logging.info(f"Function Evals: {self.function_evals}")
        self.iteration += 1
        
    def rank_select(self, population: list[dict]) -> list[dict]:
        """
        Rank-based selection, select individuals with probability proportional to their rank.
        """
        if self.problem_type == "black_box":
            population = [individual for individual in population if individual["exec_success"] and individual["obj"] < self.seed_ind["obj"]]
        else:
            population = [individual for individual in population if individual["exec_success"]]
        if len(population) < 2:
            return None
        # Sort population by objective value
        population = sorted(population, key=lambda x: x["obj"])
        ranks = [i for i in range(len(population))]
        probs = [1 / (rank + 1 + len(population)) for rank in ranks]
        # Normalize probabilities
        probs = [prob / sum(probs) for prob in probs]
        selected_population = []
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False, p=probs)
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population
    
    
    def random_select(self, population: list[dict]) -> list[dict]:
        """
        Random selection, select individuals with equal probability.
        """
        selected_population = []
        # Eliminate invalid individuals
        if self.problem_type == "black_box":
            population = [individual for individual in population if individual["exec_success"] and individual["obj"] < self.seed_ind["obj"]]
        else:
            population = [individual for individual in population if individual["exec_success"]]
        if len(population) < 2:
            return None
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False)
            # If two parents have the same objective value, consider them as identical; otherwise, add them to the selected population
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population

    def lineage_aware_select(self, population: list[dict]) -> list[dict]:
        """
        基于谱系的选择策略：结合精英选择、谱系选择和多样性维护
        
        选择策略：
        - elite_ratio: 直接选择当前最优个体
        - lineage_ratio: 根据CMP分数选择有潜力的谱系代表
        - revival_ratio: 基于多样性选择，给失败谱系机会
        """
        # 过滤有效个体
        if self.problem_type == "black_box":
            valid_pop = [ind for ind in population if ind.get("exec_success", False) and ind["obj"] < self.seed_ind["obj"]]
        else:
            valid_pop = [ind for ind in population if ind.get("exec_success", False)]
        
        if len(valid_pop) < 2:
            logging.warning("Not enough valid individuals for lineage-aware selection")
            return None
        
        # 更新所有个体的谱系统计
        for ind in valid_pop:
            if 'id' in ind:
                self.lineage_tracker.update_clade_stats(ind['id'])
        
        target_size = 2 * self.cfg.pop_size
        
        # 1. 精英选择
        elite_count = max(1, int(target_size * self.elite_ratio))
        sorted_pop = sorted(valid_pop, key=lambda x: x["obj"])
        elite_selected = sorted_pop[:elite_count]
        logging.info(f"Elite selection: {elite_count} individuals, best obj: {elite_selected[0]['obj']:.6f}")
        
        # 2. 谱系选择（基于CMP分数）
        lineage_count = max(1, int(target_size * self.lineage_ratio))
        # 按谱系分数排序
        lineage_sorted = sorted(
            valid_pop, 
            key=lambda x: self.lineage_tracker.compute_clade_score(
                x['id'], 
                self.clade_score_alpha, 
                self.clade_score_beta, 
                self.clade_score_gamma
            ) if 'id' in x else float('inf')
        )
        lineage_selected = lineage_sorted[:lineage_count]
        
        # 记录谱系选择的统计信息
        if lineage_selected and 'id' in lineage_selected[0]:
            best_clade_score = self.lineage_tracker.compute_clade_score(
                lineage_selected[0]['id'],
                self.clade_score_alpha,
                self.clade_score_beta,
                self.clade_score_gamma
            )
            logging.info(f"Lineage selection: {lineage_count} individuals, best clade score: {best_clade_score:.6f}")
        
        # 3. 复活/多样性选择
        revival_count = max(1, int(target_size * self.revival_ratio))
        already_selected_ids = set([ind.get('id') for ind in elite_selected + lineage_selected if 'id' in ind])
        revival_candidates = [ind for ind in valid_pop if ind.get('id') not in already_selected_ids]
        
        if revival_candidates:
            # 基于多样性和谱系潜力选择
            revival_scores = []
            for ind in revival_candidates:
                if 'id' in ind:
                    clade_score = self.lineage_tracker.compute_clade_score(
                        ind['id'],
                        self.clade_score_alpha,
                        self.clade_score_beta,
                        self.clade_score_gamma
                    )
                    diversity_bonus = ind.get('clade_diversity', 0.0) * 0.5
                    revival_scores.append((ind, clade_score + diversity_bonus))
                else:
                    revival_scores.append((ind, float('inf')))
            
            revival_sorted = sorted(revival_scores, key=lambda x: x[1])
            revival_selected = [x[0] for x in revival_sorted[:revival_count]]
            logging.info(f"Revival selection: {len(revival_selected)} individuals for diversity")
        else:
            revival_selected = []
            logging.info("No revival candidates available")
        
        # 合并选择结果
        selected = elite_selected + lineage_selected + revival_selected
        
        # 去重（基于ID）
        seen_ids = set()
        unique_selected = []
        for ind in selected:
            ind_id = ind.get('id')
            if ind_id and ind_id not in seen_ids:
                seen_ids.add(ind_id)
                unique_selected.append(ind)
        
        # 如果数量不足，从精英中补充
        while len(unique_selected) < target_size and len(sorted_pop) > 0:
            for ind in sorted_pop:
                if ind.get('id') not in seen_ids:
                    unique_selected.append(ind)
                    seen_ids.add(ind.get('id'))
                    if len(unique_selected) >= target_size:
                        break
            break
        
        # 确保至少有2个个体用于交叉
        if len(unique_selected) < 2:
            logging.warning("Not enough unique individuals after lineage selection")
            return None
        
        result = unique_selected[:target_size]
        logging.info(f"Total selected: {len(result)} individuals")
        
        return result

    def gen_short_term_reflection_prompt(self, ind1: dict, ind2: dict) -> tuple[list[dict], str, str]:
        """
        Short-term reflection before crossovering two individuals.
        """
        if ind1["obj"] == ind2["obj"]:
            print(ind1["code"], ind2["code"])
            raise ValueError("Two individuals to crossover have the same objective value!")
        # Determine which individual is better or worse
        if ind1["obj"] < ind2["obj"]:
            better_ind, worse_ind = ind1, ind2
        else: # robust in rare cases where two individuals have the same objective value
            better_ind, worse_ind = ind2, ind1

        worse_code = filter_code(worse_ind["code"])
        better_code = filter_code(better_ind["code"])
        
        system = self.system_reflector_prompt
        user = self.user_reflector_st_prompt.format(
            func_name = self.func_name,
            func_desc = self.func_desc,
            problem_desc = self.problem_desc,
            worse_code=worse_code,
            better_code=better_code
            )
        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # Print reflection prompt for the first iteration
        if self.print_short_term_reflection_prompt:
                logging.info("Short-term Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                self.print_short_term_reflection_prompt = False
        return message, worse_code, better_code


    def short_term_reflection(self, population: list[dict]) -> tuple[list[list[dict]], list[str], list[str]]:
        """
        Short-term reflection before crossovering two individuals.
        """
        messages_lst = []
        worse_code_lst = []
        better_code_lst = []
        # Handle odd-sized populations by skipping the last unpaired individual
        for i in range(0, len(population) - 1, 2):
            # Select two individuals
            parent_1 = population[i]
            parent_2 = population[i+1]
            
            # Short-term reflection
            messages, worse_code, better_code = self.gen_short_term_reflection_prompt(parent_1, parent_2)
            messages_lst.append(messages)
            worse_code_lst.append(worse_code)
            better_code_lst.append(better_code)
        
        # Asynchronously generate responses
        response_lst = self.short_reflector_llm.multi_chat_completion(messages_lst)
        return response_lst, worse_code_lst, better_code_lst
    
    def long_term_reflection(self, short_term_reflections: list[str]) -> None:
        """
        Long-term reflection before mutation.
        """
        system = self.system_reflector_prompt
        user = self.user_reflector_lt_prompt.format(
            problem_desc = self.problem_desc,
            prior_reflection = self.long_term_reflection_str,
            new_reflection = "\n".join(short_term_reflections),
            )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        if self.print_long_term_reflection_prompt:
            logging.info("Long-term Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_long_term_reflection_prompt = False
        
        self.long_term_reflection_str = self.long_reflector_llm.multi_chat_completion([messages])[0]
        
        # Write reflections to file
        file_name = f"problem_iter{self.iteration}_short_term_reflections.txt"
        with open(file_name, 'w') as file:
            file.writelines("\n".join(short_term_reflections) + '\n')
        
        file_name = f"problem_iter{self.iteration}_long_term_reflection.txt"
        with open(file_name, 'w') as file:
            file.writelines(self.long_term_reflection_str + '\n')


    def crossover(self, short_term_reflection_tuple: tuple[list[list[dict]], list[str], list[str]]) -> list[dict]:
        reflection_content_lst, worse_code_lst, better_code_lst = short_term_reflection_tuple
        messages_lst = []
        for reflection, worse_code, better_code in zip(reflection_content_lst, worse_code_lst, better_code_lst):
            # Crossover
            system = self.system_generator_prompt
            func_signature0 = self.func_signature.format(version=0)
            func_signature1 = self.func_signature.format(version=1)
            user = self.crossover_prompt.format(
                user_generator = self.user_generator_prompt,
                func_signature0 = func_signature0,
                func_signature1 = func_signature1,
                worse_code = worse_code,
                better_code = better_code,
                reflection = reflection,
                func_name = self.func_name,
            )
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            messages_lst.append(messages)
            
            # Print crossover prompt for the first iteration
            if self.print_crossover_prompt:
                logging.info("Crossover Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                self.print_crossover_prompt = False
        
        # Asynchronously generate responses
        response_lst = self.crossover_llm.multi_chat_completion(messages_lst)
        crossed_population = [self.response_to_individual(response, response_id) for response_id, response in enumerate(response_lst)]

        assert len(crossed_population) == len(messages_lst), f"Expected {len(messages_lst)} offspring, got {len(crossed_population)}"
        return crossed_population


    def mutate(self) -> list[dict]:
        """Elitist-based mutation. We only mutate the best individual to generate n_pop new individuals."""
        system = self.system_generator_prompt
        func_signature1 = self.func_signature.format(version=1) 
        user = self.mutation_prompt.format(
            user_generator = self.user_generator_prompt,
            reflection = self.long_term_reflection_str + self.external_knowledge,
            func_signature1 = func_signature1,
            elitist_code = filter_code(self.elitist["code"]),
            func_name = self.func_name,
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.print_mutate_prompt:
            logging.info("Mutation Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_mutate_prompt = False
        responses = self.mutation_llm.multi_chat_completion([messages], int(self.cfg.pop_size * self.mutation_rate))
        population = [self.response_to_individual(response, response_id) for response_id, response in enumerate(responses)]
        return population

    def revival_mutate(self, individual: dict) -> dict:
        """
        对未被选中的个体进行强化变异，给予复活机会
        
        与常规变异的区别：
        1. 使用谱系上下文信息指导变异
        2. 鼓励更大胆的探索（BOLD mutation）
        3. 从谱系历史中学习
        """
        system = self.system_generator_prompt
        func_signature1 = self.func_signature.format(version=1)
        
        # 获取谱系上下文
        lineage_context = ""
        if 'id' in individual:
            lineage_context = self.lineage_tracker.get_lineage_context(individual['id'])
        
        # 构建复活变异提示
        user = f"""This individual was not selected in the current generation, but shows potential based on its lineage history.

{lineage_context}

Current Code:
```python
{filter_code(individual['code'])}
```

Task: Generate a BOLD mutation that explores significantly different approaches while learning from the lineage history.
Make LARGER changes than typical mutation to give this individual a real chance at revival.
Consider:
1. What made the best ancestors in this lineage successful?
2. What patterns should be avoided based on lineage performance?
3. How can you introduce novel ideas while maintaining the core strengths?

{self.user_generator_prompt}

Reflection and Knowledge:
{self.long_term_reflection_str}
{self.external_knowledge}

Function Signature:
{func_signature1}

Generate a significantly improved version with bold innovations:
"""
        
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # 生成复活变异
        response = self.mutation_llm.multi_chat_completion([messages], 1)[0]
        
        # 创建新个体
        revival_ind = self.response_to_individual(response, f"revival_{individual.get('response_id', 'unknown')}")
        
        return revival_ind


    def evolve(self):
        while self.function_evals < self.cfg.max_fe:
            # If all individuals are invalid, stop
            if all([not individual["exec_success"] for individual in self.population]):
                raise RuntimeError(f"All individuals are invalid. Please check the stdout files in {os.getcwd()}.")
            
            # Select - 使用谱系感知选择或传统选择
            population_to_select = self.population if (self.elitist is None or self.elitist in self.population) else [self.elitist] + self.population
            
            if self.enable_lineage_selection:
                logging.info("=== Using Lineage-Aware Selection ===")
                selected_population = self.lineage_aware_select(population_to_select)
                
                # 复活机制：对未被选中的个体
                if selected_population is not None:
                    selected_ids = set([ind.get('id') for ind in selected_population if 'id' in ind])
                    unselected = [ind for ind in population_to_select 
                                 if ind.get('exec_success', False) and ind.get('id') not in selected_ids]
                    
                    # 选择部分未被选中的个体进行复活变异
                    revival_count = min(len(unselected), max(1, int(self.cfg.pop_size * self.revival_ratio)))
                    if unselected and revival_count > 0:
                        import random
                        revival_candidates = random.sample(unselected, revival_count)
                        logging.info(f"=== Revival Mechanism: {len(revival_candidates)} candidates ===")
                        
                        revival_population = []
                        for idx, ind in enumerate(revival_candidates):
                            try:
                                revival_ind = self.revival_mutate(ind)
                                revival_population.append(revival_ind)
                            except Exception as e:
                                logging.warning(f"Revival mutation failed for individual {idx}: {e}")
                        
                        if revival_population:
                            # 评估复活个体
                            revival_population = self.evaluate_population(revival_population)
                            
                            # 注册复活个体到谱系追踪器
                            for revival_ind in revival_population:
                                if revival_ind.get('exec_success', False):
                                    self.lineage_tracker.register_individual(revival_ind, parents=[ind])
                            
                            logging.info(f"Revival: {len(revival_population)} individuals created, "
                                       f"{sum(1 for ind in revival_population if ind.get('exec_success', False))} successful")
            else:
                logging.info("=== Using Traditional Random Selection ===")
                selected_population = self.random_select(population_to_select)
            
            if selected_population is None:
                raise RuntimeError("Selection failed. Please check the population.")
            
            # Short-term reflection
            short_term_reflection_tuple = self.short_term_reflection(selected_population)
            
            # Crossover
            crossed_population = self.crossover(short_term_reflection_tuple)
            
            # 注册交叉后的个体到谱系追踪器
            if self.enable_lineage_selection:
                for idx, new_ind in enumerate(crossed_population):
                    # 交叉产生的个体有两个父代
                    parent_idx = idx * 2
                    if parent_idx + 1 < len(selected_population):
                        parents = [selected_population[parent_idx], selected_population[parent_idx + 1]]
                        self.lineage_tracker.register_individual(new_ind, parents=parents)
            
            # Evaluate
            self.population = self.evaluate_population(crossed_population)
            
            # Update
            self.update_iter()
            
            # 记录谱系统计
            if self.enable_lineage_selection and self.iteration % 5 == 0:
                self.lineage_tracker.log_statistics()
            
            # Long-term reflection
            self.long_term_reflection([response for response in short_term_reflection_tuple[0]])
            
            # Mutate
            mutated_population = self.mutate()
            
            # 注册变异后的个体到谱系追踪器
            if self.enable_lineage_selection and self.elitist is not None:
                for mut_ind in mutated_population:
                    self.lineage_tracker.register_individual(mut_ind, parents=[self.elitist])
            
            # Evaluate
            self.population.extend(self.evaluate_population(mutated_population))
            
            # Update
            self.update_iter()

        return self.best_code_overall, self.best_code_path_overall
