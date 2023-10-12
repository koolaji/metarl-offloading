import numpy as np
import logging
from utils import logger


class MTLBO:
    def __init__(self, population_size, num_generations, policy, env, sampler, sampler_processor, generation):
        self.population_size = population_size
        self.num_generations = num_generations
        self.policy = policy
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sampler_processor
        self.generation = 0

    # def objective_function(self, policy_params):
    #     self.policy.set_params(policy_params)
    #     paths = self.sampler.obtain_samples()
    #     samples_data = self.sampler_processor.process_samples(paths)
    #     return -np.mean(samples_data['rewards'])

    def teacher_phase(self, population):
        logging.info('teacher_phase')
        # Compute the mean solution
        mean_solution = {}
        keys = population[0].keys()
        for key in keys:
            mean_solution[key] = np.mean([individual[key] for individual in population], axis=0)

        # Update the solutions using sine and cosine functions
        for student in population:
            diff = self.subtract_dicts(student, mean_solution)
            
            # Use sine or cosine based on a random choice
            if np.random.rand() < 0.5:
                scaled_diff = self.scale_dict(diff, np.random.random() * np.sin(np.random.uniform(-np.pi/2, np.pi/2)))
            else:
                scaled_diff = self.scale_dict(diff, np.random.random() * np.cos(np.random.uniform(0, np.pi)))
            
            new_solution = self.add_dicts(student, scaled_diff)
            logging.info('teacher_phase')
            if self.objective_function(new_solution) < self.objective_function(student):
                student = new_solution


    def subtract_dicts(self,dict1, dict2):
        result = {}
        for key in dict1:
            result[key] = dict1[key] - dict2[key]
        return result

    def scale_dict(self,d, scalar):
            """Multiply each value in the dictionary by a scalar."""
            return {key: d[key] * scalar for key in d}
    def add_dicts(self,dict1, dict2):
            """Add values in two dictionaries with the same keys."""
            return {key: dict1[key] + dict2[key] for key in dict1}

    def learner_phase(self, population):
        for i in range(self.population_size):
            logging.info('learner_phase %s', i)
            j = np.random.choice([x for x in range(self.population_size) if x != i])
            logging.info('learner_phase random')
            diff = self.subtract_dicts(population[i], population[j])

            # Use sine or cosine based on a random choice
            if np.random.rand() < 0.5:
                scaled_diff = self.scale_dict(diff, np.random.random() * np.sin(np.random.uniform(-np.pi/2, np.pi/2)))
            else:
                scaled_diff = self.scale_dict(diff, np.random.random() * np.cos(np.random.uniform(0, np.pi)))

            logging.info('learner_phase scaled_diff')
            new_solution = self.add_dicts(population[i], scaled_diff)
            logging.info(f'learner_phase new_solution {i}')
            if self.objective_function(new_solution) < self.objective_function(population[i]):
                population[i] = new_solution
            logging.info('learner_phase objective_function')
       

    def optimize(self):
        logging.info('optimize ')
        population = [self.policy.get_random_params() for _ in range(self.population_size)]        
        for generation in range(self.num_generations):
            self.generation = generation
            logging.info('teacher_phase ')
            self.teacher_phase(population)
            logging.info('optimize ')
            self.learner_phase(population)

            """ ------------------- Logging Stuff --------------------------"""
            new_paths = self.sampler.obtain_samples(log=True, log_prefix='')
            new_samples_data = self.sampler_processor.process_samples(new_paths, log="all", log_prefix='')
            ret = np.array([])
            logging.info('optimize len new_samples_data = %s', len(new_samples_data))
            for i in range(1):
                ret = np.concatenate((ret, np.sum(new_samples_data[i]['rewards'], axis=-1)), axis=-1)

            avg_reward = np.mean(ret)

            latency = np.array([])
            for i in range(1):
                latency = np.concatenate((latency, new_samples_data[i]['finish_time']), axis=-1)

            avg_latency = np.mean(latency)

            logger.logkv('generation', generation)
            logger.logkv('Average reward, ', avg_reward)
            logger.logkv('Average latency,', "{:.4f}".format(avg_latency))

            logger.dumpkvs()
        
        # Return the best solution found
        return min(population, key=self.objective_function)
    
    def objective_function(self, policy_params):
                logging.debug('objective_function')
                self.policy.set_params(policy_params)
                paths = self.sampler.obtain_samples(log=False, log_prefix='')
                samples_data = self.sampler_processor.process_samples(paths, log="all", log_prefix='')
                ret = np.array([])
                logging.info('objective_function len samples_data = %s', len(samples_data))
                for i in range(1):
                    ret = np.concatenate((ret, np.sum(samples_data[i]['rewards'], axis=-1)), axis=-1)

                avg_reward = np.mean(ret)

                latency = np.array([])
                for i in range(1):
                    latency = np.concatenate((latency, samples_data[i]['finish_time']), axis=-1)

                avg_latency = np.mean(latency)

                logger.logkv('generation', self.generation)
                logger.logkv('Average reward, ', avg_reward)
                logger.logkv('Average latency,', "{:.4f}".format(avg_latency))

                logger.dumpkvs()
                # Assuming samples_data is a list of dictionaries
                all_latency = [data['finish_time'] for data in samples_data]
                latency = np.concatenate(all_latency, axis=0)
                
                return -np.mean(latency)