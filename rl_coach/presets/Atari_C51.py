from rl_coach.agents.categorical_dqn_agent import CategoricalDQNAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.environments.gym_environment import Atari, atari_deterministic_v4, atari_schedule
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager

#########
# Agent #
#########
agent_params = CategoricalDQNAgentParameters()
agent_params.network_wrappers['main'].learning_rate = 0.01

###############
# Environment #
###############
env_params = Atari(level=SingleLevelSelection(atari_deterministic_v4))

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.trace_test_levels = ['breakout', 'pong', 'space_invaders']

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=atari_schedule, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
