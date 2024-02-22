from models.feature_extractor import DefaultCNNFeatureExtractor,\
                                    CuriosityLikeCNNFeatureExtractor

from models.inverse_module import CuriosityLinearInverseModule,\
                                Layer3LinearInverseModule

from models.feature_predictor import CuriosityLinearFeaturePredictor

from models.inner_state_predictor import DefaultLSTMInnerStatePredictor

from models.controller import DefaultDiscreteLinearActorCritic,\
                            LSTMDiscreteLinearActorCritic