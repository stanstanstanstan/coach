from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters
from rl_coach.agents.soft_actor_critic_agent import SoftActorCriticAgentParameters
from rl_coach.architectures.layers import Dense
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.carla_environment import CarlaEnvironmentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import LinearSchedule
from rl_coach.memories.memory import MemoryGranularity

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(20)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(1000)

#########
# Agent #
#########

agent_params = SoftActorCriticAgentParameters()
# override default parameters:
# value (v) networks parameters
agent_params.network_wrappers['v'].batch_size = 32
agent_params.network_wrappers['v'].learning_rate = 0.0003
agent_params.network_wrappers['v'].middleware_parameters.scheme = [Dense(32)]
agent_params.network_wrappers['v'].optimizer_epsilon = 1e-5
agent_params.network_wrappers['v'].adam_optimizer_beta2 = 0.999
agent_params.network_wrappers['v'].input_embedders_parameters['forward_camera'] = \
    agent_params.network_wrappers['v'].input_embedders_parameters.pop('observation')

# critic (q) network parameters
agent_params.network_wrappers['q'].heads_parameters[0].network_layers_sizes = (32, 32)
agent_params.network_wrappers['q'].batch_size = 32
agent_params.network_wrappers['q'].learning_rate = 0.0003
agent_params.network_wrappers['q'].optimizer_epsilon = 1e-5
agent_params.network_wrappers['q'].adam_optimizer_beta2 = 0.999
agent_params.network_wrappers['q'].input_embedders_parameters['forward_camera'] = \
    agent_params.network_wrappers['q'].input_embedders_parameters.pop('observation')

# actor (policy) network parameters
agent_params.network_wrappers['policy'].batch_size = 32
agent_params.network_wrappers['policy'].learning_rate = 0.0003
agent_params.network_wrappers['policy'].middleware_parameters.scheme = [Dense(32)]
agent_params.network_wrappers['policy'].optimizer_epsilon = 1e-5
agent_params.network_wrappers['policy'].adam_optimizer_beta2 = 0.999
agent_params.network_wrappers['policy'].input_embedders_parameters['forward_camera'] = \
    agent_params.network_wrappers['policy'].input_embedders_parameters.pop('observation')


###############
# Environment #
###############
env_params = CarlaEnvironmentParameters(level='town2')

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters())