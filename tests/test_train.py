# from omegaconf import DictConfig
# import auem.train


# def test_binary_classification_with_random_data():
#     dataset_config = {
#         "_target_": "traincore.data.sets.random.RandomAudioWithClassifierDataset",
#         "n_samples": 22050,
#         "n_classes": 2,
#     }
#     config = DictConfig(
#         {
#             "logger": None,
#             "data": {
#                 "_target_": "traincore.data.modules.generic.BasicDataModule",
#                 "datasets": {
#                     "train": {"n_examples": 20, **dataset_config},
#                     "validation": {"n_examples": 5, **dataset_config},
#                     "test": None,
#                 },
#             },
#             "model": {
#                 "_target_": "traincore.models.simplecnn.SimpleCNNBase",
#                 "encoder": {
#                     "_target_": "traincore.models.encoders.melspec.MelEncoder",
#                     "n_mels": 80,
#                 },
#                 # "input_shape": (1, 1, 80, 11),
#                 "num_classes": 2,
#             },
#             "recipe": None,
#             "trainer": {"_target_": "lightning.pytorch.trainer.trainer.Trainer"},
#         }
#     )

#     auem.train.train(config)
