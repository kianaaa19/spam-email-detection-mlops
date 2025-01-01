#!/usr/bin/env python
# coding: utf-8

# # Install library requirements

# In[1]:


# Install
get_ipython().system('pip install tfx')
get_ipython().system('pip install opencv-python')
get_ipython().system('pip install -q kaggle')
get_ipython().system('sudo install docker')


# 
# 
# ---
# 
# 
# 
# ---
# 
# 
# 
# ---
# 
# 

# # WRITE FILE: COMPONENTS.PY

# 

# In[2]:


COMPONENTS = "spam_components.py"


# In[3]:


get_ipython().run_cell_magic('writefile', '{COMPONENTS}', '"""\nspam_components.py\n"""\nimport os\n\nimport tensorflow_model_analysis as tfma\nfrom tfx.components import (CsvExampleGen, Evaluator, ExampleValidator, Pusher,\n                            SchemaGen, StatisticsGen, Trainer, Transform,\n                            Tuner)\nfrom tfx.dsl.components.common.resolver import Resolver\nfrom tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import \\\n    LatestBlessedModelStrategy\nfrom tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2\nfrom tfx.types import Channel\nfrom tfx.types.standard_artifacts import Model, ModelBlessing\n\n\ndef init_components(\n    data_dir,\n    transform_module,\n    trainer_module,\n    tuner_module,\n    train_steps,\n    eval_steps,\n    serving_model_dir,\n):\n    """Initiate tfx pipeline components\n\n    Args:\n        args (dict): args that containts some dependencies\n\n    Returns:\n        tuple: TFX pipeline components\n    """\n    output = example_gen_pb2.Output(\n        split_config=example_gen_pb2.SplitConfig(splits=[\n            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),\n            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),\n        ])\n    )\n\n    example_gen = CsvExampleGen(\n        input_base=data_dir,\n        output_config=output,\n    )\n\n    statistics_gen = StatisticsGen(\n        examples=example_gen.outputs["examples"],\n    )\n\n    schema_gen = SchemaGen(\n        statistics=statistics_gen.outputs["statistics"],\n    )\n\n    example_validator = ExampleValidator(\n        statistics=statistics_gen.outputs["statistics"],\n        schema=schema_gen.outputs["schema"],\n    )\n\n    transform = Transform(\n        examples=example_gen.outputs["examples"],\n        schema=schema_gen.outputs["schema"],\n        module_file=os.path.abspath(transform_module),\n    )\n\n    tuner = Tuner(\n        module_file=os.path.abspath(tuner_module),\n        examples=transform.outputs["transformed_examples"],\n        transform_graph=transform.outputs["transform_graph"],\n        schema=schema_gen.outputs["schema"],\n        train_args=trainer_pb2.TrainArgs(\n            splits=["train"],\n            num_steps=train_steps,\n        ),\n        eval_args=trainer_pb2.EvalArgs(\n            splits=["eval"],\n            num_steps=eval_steps,\n        ),\n    )\n\n    trainer = Trainer(\n        module_file=trainer_module,\n        examples=transform.outputs["transformed_examples"],\n        transform_graph=transform.outputs["transform_graph"],\n        schema=schema_gen.outputs["schema"],\n        hyperparameters=tuner.outputs["best_hyperparameters"],\n        train_args=trainer_pb2.TrainArgs(\n            splits=["train"],\n            num_steps=train_steps,\n        ),\n        eval_args=trainer_pb2.EvalArgs(\n            splits=["eval"],\n            num_steps=eval_steps\n        ),\n    )\n\n    model_resolver = Resolver(\n        strategy_class=LatestBlessedModelStrategy,\n        model=Channel(type=Model),\n        model_blessing=Channel(type=ModelBlessing),\n    ).with_id("Latest_blessed_model_resolve")\n\n    eval_config = tfma.EvalConfig(\n        model_specs=[tfma.ModelSpec(label_key="label")],\n        slicing_specs=[\n            tfma.SlicingSpec(),\n            tfma.SlicingSpec(feature_keys=["text"]),\n        ],\n        metrics_specs=[\n            tfma.MetricsSpec(metrics=[\n                tfma.MetricConfig(class_name="AUC"),\n                tfma.MetricConfig(class_name="Precision"),\n                tfma.MetricConfig(class_name="Recall"),\n                tfma.MetricConfig(class_name="ExampleCount"),\n                tfma.MetricConfig(class_name="TruePositives"),\n                tfma.MetricConfig(class_name="FalsePositives"),\n                tfma.MetricConfig(class_name="TrueNegatives"),\n                tfma.MetricConfig(class_name="FalseNegatives"),\n                tfma.MetricConfig(\n                    class_name="BinaryAccuracy",\n                    threshold=tfma.MetricThreshold(\n                        value_threshold=tfma.GenericValueThreshold(\n                            lower_bound={"value": .6},\n                        ),\n                        change_threshold=tfma.GenericChangeThreshold(\n                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,\n                            absolute={"value": 1e-4},\n                        ),\n                    ),\n                ),\n            ]),\n        ],\n    )\n\n    evaluator = Evaluator(\n        examples=example_gen.outputs["examples"],\n        model=trainer.outputs["model"],\n        baseline_model=model_resolver.outputs["model"],\n        eval_config=eval_config,\n    )\n\n    pusher = Pusher(\n        model=trainer.outputs["model"],\n        model_blessing=evaluator.outputs["blessing"],\n        push_destination=pusher_pb2.PushDestination(\n            filesystem=pusher_pb2.PushDestination.Filesystem(\n                base_directory=serving_model_dir,\n            ),\n        ),\n    )\n\n    return (\n        example_gen,\n        statistics_gen,\n        schema_gen,\n        example_validator,\n        transform,\n        tuner,\n        trainer,\n        model_resolver,\n        evaluator,\n        pusher,\n    )\n')


# # WRITE FILE: PIPELINE.PY

# In[4]:


PIPELINE = "spam_pipeline.py"


# In[28]:


get_ipython().run_cell_magic('writefile', '{PIPELINE}', '"""\nspam_pipeline.py\n"""\nfrom typing import Text\n\nfrom absl import logging\nfrom tfx.orchestration import metadata, pipeline\n\n\ndef init_pipeline(pipeline_root: Text, pipeline_name, metadata_path, components):\n    """Initiate tfx pipeline\n\n    Args:\n        pipeline_root (Text): a path to th pipeline directory\n        pipeline_name (str): pipeline name\n        metadata_path (str): a path to the metadata directory\n        components (dict): tfx components\n\n    Returns:\n        pipeline.Pipeline: pipeline orchestration\n    """\n\n    logging.info(f"Pipeline root set to: {pipeline_root}")\n\n    beam_args = [\n        "--direct_running_mode=multi_processing",\n        "----direct_num_workers=0",\n    ]\n\n    return pipeline.Pipeline(\n        pipeline_name=pipeline_name,\n        pipeline_root=pipeline_root,\n        components=components,\n        enable_cache=True,\n        metadata_connection_config=metadata.sqlite_metadata_connection_config(\n            metadata_path,\n        ),\n        eam_pipeline_args=beam_args,\n    )\n')


# # WRITE FILE: TRANSFORM.PY
# Processing yang dilakukan hanya untuk merubah nama feature dan label yang telah di transform menjadi label_xf, text_xf, dan transform feature kedalam format lowercase string dan untuk casting label kedalam format int64 untuk memastikan tipe data pada label

# In[7]:


TRANSFORM_MODULE_FILE = "spam_transform.py"


# In[8]:


get_ipython().run_cell_magic('writefile', '{TRANSFORM_MODULE_FILE}', '"""\nspam_transform.py\n"""\nimport tensorflow as tf\nLABEL_KEY = "label"\nFEATURE_KEY = "text"\ndef transformed_name(key):\n    """Renaming transformed features"""\n    return key + "_xf"\ndef preprocessing_fn(inputs):\n    """\n    Preprocess input features into transformed features\n\n    Args:\n        inputs: map from feature keys to raw features.\n\n    Return:\n        outputs: map from feature keys to transformed features.\n    """\n\n    outputs = {}\n\n    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])\n\n    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)\n\n    return outputs\n')


# # WRITE FILE: TUNER.PY
# Melakukan tuning hyperparameter pada arsitektur model yang digunakan untuk mendapatkan hyperparameter terbaik.

# In[9]:


TUNER_FILE = "tuner_module_file.py"


# In[10]:


get_ipython().run_cell_magic('writefile', '{TUNER_FILE}', '"""\ntuner_module_file.py\n"""\nfrom typing import NamedTuple, Dict, Text, Any\nimport keras_tuner as kt\nimport tensorflow as tf\nimport tensorflow_transform as tft\nfrom keras_tuner.engine import base_tuner\nfrom tfx.components.trainer.fn_args_utils import FnArgs\n\n\nLABEL_KEY = "label"\nFEATURE_KEY = "text"\nNUM_EPOCHS = 1\n\nTunerFnResult = NamedTuple("TunerFnResult", [\n    ("tuner", base_tuner.BaseTuner),\n    ("fit_kwargs", Dict[Text, Any]),\n])\n\nearly_stopping_callback = tf.keras.callbacks.EarlyStopping(\n    monitor="val_binary_accuracy",\n    mode="max",\n    verbose=1,\n    patience=10,\n)\n\n\ndef transformed_name(key):\n    return f"{key}_xf"\n\n\ndef gzip_reader_fn(filenames):\n    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")\n\n\ndef input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=8):\n    transform_feature_spec = (\n        tf_transform_output.transformed_feature_spec().copy()\n    )\n\n    dataset = tf.data.experimental.make_batched_features_dataset(\n        file_pattern=file_pattern,\n        batch_size=batch_size,\n        features=transform_feature_spec,\n        reader=gzip_reader_fn,\n        num_epochs=num_epochs,\n        label_key=transformed_name(LABEL_KEY),\n    )\n\n    return dataset\n\n\ndef model_builder(hp, vectorizer_layer):\n    num_hidden_layers = hp.Choice(\n        "num_hidden_layers", values=[1, 2]\n    )\n    embed_dims = hp.Int(\n        "embed_dims", min_value=16, max_value=128, step=32\n    )\n    lstm_units= hp.Int(\n        "lstm_units", min_value=32, max_value=128, step=32\n    )\n    dense_units = hp.Int(\n        "dense_units", min_value=32, max_value=256, step=32\n    )\n    dropout_rate = hp.Float(\n        "dropout_rate", min_value=0.1, max_value=0.5, step=0.1\n    )\n    learning_rate = hp.Choice(\n        "learning_rate", values=[1e-2, 1e-3, 1e-4]\n    )\n\n    inputs = tf.keras.Input(\n        shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string\n    )\n\n    x = vectorizer_layer(inputs)\n    x = tf.keras.layers.Embedding(input_dim=5000, output_dim=embed_dims)(x)\n    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units))(x)\n\n    for _ in range(num_hidden_layers):\n        x = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(x)\n        x = tf.keras.layers.Dropout(dropout_rate)(x)\n\n    outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)\n\n    model = tf.keras.Model(inputs=inputs, outputs=outputs)\n\n    model.compile(\n        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),\n        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),\n        metrics=["binary_accuracy"],\n    )\n\n    return model\n\n\ndef tuner_fn(fn_args: FnArgs):\n    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)\n\n    train_set = input_fn(\n        fn_args.train_files[0], tf_transform_output, NUM_EPOCHS\n    )\n    eval_set = input_fn(\n        fn_args.eval_files[0], tf_transform_output, NUM_EPOCHS\n    )\n\n    vectorizer_dataset = train_set.map(\n        lambda f, l: f[transformed_name(FEATURE_KEY)]\n    )\n\n    vectorizer_layer = tf.keras.layers.TextVectorization(\n        max_tokens=5000,\n        output_mode="int",\n        output_sequence_length=500,\n    )\n    vectorizer_layer.adapt(vectorizer_dataset)\n\n    tuner = kt.Hyperband(\n        hypermodel=lambda hp: model_builder(hp, vectorizer_layer),\n        objective=kt.Objective(\'binary_accuracy\', direction=\'max\'),\n        max_epochs=NUM_EPOCHS,\n        factor=3,\n        directory=fn_args.working_dir,\n        project_name="kt_hyperband",\n    )\n\n    return TunerFnResult(\n        tuner=tuner,\n        fit_kwargs={\n            "callbacks": [early_stopping_callback],\n            "x": train_set,\n            "validation_data": eval_set,\n        },\n    )\n')


# # WRITE FILE: TRAINER.PY
# Melatih model dengan menggunakan hyperparameter yang telah dituning sebelumnya

# In[11]:


TRAINER = "spam_trainer.py"


# In[12]:


get_ipython().run_cell_magic('writefile', '{TRAINER}', '"""\nspam_trainer.py\n"""\nimport os\nimport tensorflow as tf\nimport tensorflow_transform as tft\nimport tensorflow_hub as hub\nfrom tfx.components.trainer.fn_args_utils import FnArgs\n\nLABEL_KEY = "label"\nFEATURE_KEY = "text"\n\ndef transformed_name(key):\n    """Renaming transformed features"""\n    return key + "_xf"\n\ndef gzip_reader_fn(filenames):\n    """Loads compressed data"""\n    return tf.data.TFRecordDataset(filenames, compression_type=\'GZIP\')\n\n\ndef input_fn(file_pattern,\n             tf_transform_output,\n             num_epochs,\n             batch_size=64)->tf.data.Dataset:\n    """Get post_tranform feature & create batches of data"""\n\n    # Get post_transform feature spec\n    transform_feature_spec = (\n        tf_transform_output.transformed_feature_spec().copy())\n\n    # create batches of data\n    dataset = tf.data.experimental.make_batched_features_dataset(\n        file_pattern=file_pattern,\n        batch_size=batch_size,\n        features=transform_feature_spec,\n        reader=gzip_reader_fn,\n        num_epochs=num_epochs,\n        label_key = transformed_name(LABEL_KEY))\n    return dataset\n\n# os.environ[\'TFHUB_CACHE_DIR\'] = \'/hub_chace\'\n# embed = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4")\n\n# Vocabulary size and number of words in a sequence.\nVOCAB_SIZE = 10000\nSEQUENCE_LENGTH = 100\n\nvectorize_layer = tf.keras.layers.TextVectorization(\n    standardize="lower_and_strip_punctuation",\n    max_tokens=VOCAB_SIZE,\n    output_mode=\'int\',\n    output_sequence_length=SEQUENCE_LENGTH)\n\n\nEMBEDDING_DIM=16\ndef model_builder():\n    """Build machine learning model"""\n    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)\n    reshaped_narrative = tf.reshape(inputs, [-1])\n    x = vectorize_layer(reshaped_narrative)\n    x = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, name="embedding")(x)\n    x = tf.keras.layers.GlobalAveragePooling1D()(x)\n    x = tf.keras.layers.Dense(64, activation=\'relu\')(x)\n    x = tf.keras.layers.Dense(32, activation="relu")(x)\n    outputs = tf.keras.layers.Dense(1, activation=\'sigmoid\')(x)\n\n\n    model = tf.keras.Model(inputs=inputs, outputs = outputs)\n\n    model.compile(\n        loss = \'binary_crossentropy\',\n        optimizer=tf.keras.optimizers.Adam(0.01),\n        metrics=[tf.keras.metrics.BinaryAccuracy()]\n\n    )\n\n    # print(model)\n    model.summary()\n    return model\n\n\ndef _get_serve_tf_examples_fn(model, tf_transform_output):\n\n    model.tft_layer = tf_transform_output.transform_features_layer()\n\n    @tf.function\n    def serve_tf_examples_fn(serialized_tf_examples):\n\n        feature_spec = tf_transform_output.raw_feature_spec()\n\n        feature_spec.pop(LABEL_KEY)\n\n        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)\n\n        transformed_features = model.tft_layer(parsed_features)\n\n        # get predictions using the transformed features\n        return model(transformed_features)\n\n    return serve_tf_examples_fn\n\ndef run_fn(fn_args: FnArgs) -> None:\n\n    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), \'logs\')\n\n    tensorboard_callback = tf.keras.callbacks.TensorBoard(\n        log_dir = log_dir, update_freq=\'batch\'\n    )\n\n    es = tf.keras.callbacks.EarlyStopping(monitor=\'val_binary_accuracy\', mode=\'max\', verbose=1, patience=10)\n    mc = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir, monitor=\'val_binary_accuracy\', mode=\'max\', verbose=1, save_best_only=True)\n\n\n    # Load the transform output\n    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)\n\n    # Create batches of data\n    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)\n    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)\n    vectorize_layer.adapt(\n        [j[0].numpy()[0] for j in [\n            i[0][transformed_name(FEATURE_KEY)]\n                for i in list(train_set)]])\n\n    # Build the model\n    model = model_builder()\n\n\n    # Train the model\n    model.fit(x = train_set,\n            validation_data = val_set,\n            callbacks = [tensorboard_callback, es, mc],\n            steps_per_epoch = 1000,\n            validation_steps= 1000,\n            epochs=10)\n    signatures = {\n        \'serving_default\':\n        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(\n                                    tf.TensorSpec(\n                                    shape=[None],\n                                    dtype=tf.string,\n                                    name=\'examples\'))\n    }\n    model.save(fn_args.serving_model_dir, save_format=\'tf\', signatures=signatures)\n')


# # WRITE FILE: DOCKERFILE

# In[13]:


DOCKER = "Dockerfile"


# In[14]:


get_ipython().run_cell_magic('writefile', '{DOCKER}', 'FROM tensorflow/serving:latest\n\nCOPY ./serving_model_dir /models\nENV MODEL_NAME=spam-detection-model\n')


# # WRITE FILE: REQUIREMENTS

# In[15]:


REQ = "requirements.txt"


# In[16]:


get_ipython().run_cell_magic('writefile', '{REQ}', "# Install\n!pip install tfx\n!pip install opencv-python\n!pip install -q kaggle\n!sudo install docker\n\n# Import dependencies\nimport pandas as pd\nimport numpy as np\nimport cv2\nimport os\nimport zipfile\nimport shutil\nimport pathlib\nimport tensorflow as tf\nimport matplotlib.pyplot as plt\nimport matplotlib.image as mpimg\nfrom google.colab import files\nfrom google.colab.patches import cv2_imshow\nfrom tensorflow.keras.models import Sequential, load_model, Model\nfrom sklearn.model_selection import train_test_split\nfrom tensorflow.keras import layers\nfrom tensorflow.keras.layers import Input, Activation, Add, Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dense, Input\nfrom tensorflow.keras.regularizers import l2\nfrom tensorflow.keras.optimizers import Adam\nimport nltk\nnltk.download('stopwords')\nfrom nltk.corpus import stopwords\nfrom nltk.tokenize import word_tokenize\nfrom sklearn.model_selection import train_test_split\nfrom tensorflow.keras.preprocessing.text import Tokenizer\nfrom tensorflow.keras.preprocessing.sequence import pad_sequences\nfrom tensorflow.keras.layers import Embedding, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D, LSTM, Flatten\nfrom tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping\nfrom sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report\nfrom tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint\nfrom tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Tuner\nfrom tfx.proto import example_gen_pb2\nfrom tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext\nimport tensorflow_hub as hub\nfrom tfx.components import Tuner\nfrom tfx.proto import trainer_pb2\nfrom tfx.dsl.components.common.resolver import Resolver\nfrom tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy\nfrom tfx.types import Channel\nfrom tfx.types.standard_artifacts import Model, ModelBlessing\nimport tensorflow_model_analysis as tfma\nfrom tfx.components import Evaluator\nfrom tfx.components import Pusher\nfrom tfx.proto import pusher_pb2\nimport tensorflow_transform as tft\n\n# Conda:\npip install pandas numpy opencv-python tensorflow matplotlib scikit-learn nltk tensorflow-hub tensorflow-model-analysis tensorflow-transform\npip install tfx\n")


# In[17]:


REQTF = "requirements-tfserving.txt"


# In[18]:


get_ipython().run_cell_magic('writefile', '{REQTF}', 'tensorflow-serving-api\ntransformers\n')


# 
# 
# ---
# 
# 
# 
# ---
# 
# 
# 
# ---
# 
# 

# # IMPORT LIBRARY

# In[20]:


import os
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from modules.spam_pipeline import init_local_pipeline
from modules.spam_components import init_components


# # SET VARIABEL

# In[21]:


PIPELINE_NAME = 'kianaa19-pipeline'

DATA_ROOT = 'data'
TRANSFORM_MODULE_FILE = 'modules/spam_transform.py'
TUNER_MODULE_FILE = 'modules/tuner_module_file.py'
TRAINER_MODULE_FILE = 'modules/spam_trainer.py'

OUTPUT_BASE = 'output'
serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, 'metadata.sqlite')


# # RUN PIPELINE

# In[ ]:


from modules.spam_pipeline1 import init_pipeline
from modules.spam_components import init_components

components = init_components(
    data_dir=DATA_ROOT,
    transform_module=TRANSFORM_MODULE_FILE,
    trainer_module=TRAINER_MODULE_FILE,
    tuner_module=TUNER_MODULE_FILE,
    train_steps=1000,
    eval_steps=800,
    serving_model_dir=serving_model_dir,
)

pipeline = init_pipeline(
    pipeline_root,
    PIPELINE_NAME,
    metadata_path,
    components
)
BeamDagRunner().run(pipeline)


# 
# 
# ---
# 
# 
# 
# ---
# 
# 
# 
# ---
# 
# 

# # PYLINT

# In[ ]:


get_ipython().system('pip install pylint')
get_ipython().system('pip install autopep8')


# In[ ]:


get_ipython().system('pylint spam_components.py')


# In[ ]:


get_ipython().system('autopep8 --in-place --aggressive --aggressive spam_components.py')


# In[ ]:


get_ipython().system('pylint spam_components.py')


# In[ ]:


get_ipython().system('pylint spam_pipeline.py')


# In[ ]:


get_ipython().system('autopep8 --in-place --aggressive --aggressive spam_pipeline.py')


# In[ ]:


get_ipython().system('pylint spam_pipeline.py')


# In[ ]:


get_ipython().system('pylint spam_transform.py')


# In[ ]:


get_ipython().system('autopep8 --in-place --aggressive --aggressive spam_transform.py')


# In[ ]:


get_ipython().system('pylint spam_transform.py')


# In[ ]:


get_ipython().system('pylint tuner_module_file.py')


# In[ ]:


get_ipython().system('autopep8 --in-place --aggressive --aggressive tuner_module_file.py')


# In[ ]:


get_ipython().system('pylint tuner_module_file.py')


# In[ ]:


get_ipython().system('pylint spam_trainer.py')


# In[ ]:


get_ipython().system('autopep8 --in-place --aggressive --aggressive spam_trainer.py')


# In[ ]:


get_ipython().system('pylint spam_trainer.py')

