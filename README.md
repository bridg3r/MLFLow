---
title: "MLflow"
author: "Brigham Davis, Bridger Hackworth, LaRena Allen"
output: 
  html_document:
    theme: sandstone
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

# {.tabset .tabset-pills}

## Why MLflow:

### What Problems Does MLflow solve?

<br/> 1. **Keep Track of Experiments**

When you are just working with files on your laptop, or with an interactive notebook, how do you tell which data, code and parameters went into getting a particular result?

<br/> 2. **Reproduce Code**

Even if you have meticulously tracked the code versions and parameters, you need to capture the whole environment (for example, library dependencies) to get the same result again. This is especially challenging if you want another data scientist to use your code, or if you want to run the same code at scale on another platform (for example, in the cloud).

<br/> 3. **Set a Standard for Packaging and Deploying Models**

Every data science team comes up with its own approach for each ML library that it uses, and the link between a model and the code and parameters that produced it is often lost.

<br/> 4. **Provide Central Storage for Model Managment**

A data science team creates many models. In absence of a central place to collaborate and manage model lifecycle, data science teams face challenges in how they manage models stages: from development to staging, and finally, to archiving or production, with respective versions, annotations, and history.


### From MLflow

*"MLflow lets you train, reuse, and deploy models with any library and package them into reproducible steps that other data scientists can use as a “black box,” without even having to know which library you are using.""*

## MLflow Components:

### Components of MLflow {.tabset}

#### MLflow Tracking

MLflow Tracking is an API and UI for logging parameters, code versions, metrics, and artifacts when running your machine learning code and for later visualizing the results. 

You can use MLflow Tracking in any environment (for example, a standalone script or a notebook) to log results to local files or to a server, then compare multiple runs.

Teams can also use it to compare results from different users.

##### MLflow Tracking in DataBricks

Locally, MLflow stores tracking data and artifacts in an **mlruns/** subdirectory of where you ran the code.

You may also store your data remotely. You can track your runs with a tracking server in a Databricks workspace. To do so:

- Call ```mlflow.set_tracking_uri``` in your code; or

- Set the ```MLFLOW_TRACKING_URI``` environment variable

A tracking server is a lightweight HTTP server built in to MLflow. You can run a tracking server on a network-accessible server by running:

```{r, eval=F, echo=T}
mlflow server
```


##### MLflow Automatic Logging

Automatic logging allows you to log metrics, parameters, and models without the need for explicit log statements.

There are two ways to use autologging:

1. Call ```mlflow.autolog()``` **before your training code**. This will enable autologging for each supported library you have installed as soon as you import it.

2. Use library-specific autolog calls for each library you use in your code. See [MLflow Documentation](https://mlflow.org/docs/latest/tracking.html#scikit-learn) for examples.

The following libraries support autologging:

- [Scikit-learn](https://mlflow.org/docs/latest/tracking.html#scikit-learn)

- [Keras](https://mlflow.org/docs/latest/tracking.html#keras)

- [Gluon](https://mlflow.org/docs/latest/tracking.html#gluon)

- [XGBoost](https://mlflow.org/docs/latest/tracking.html#xgboost)

- [LightGBM](https://mlflow.org/docs/latest/tracking.html#lightgbm)

- [Statsmodels](https://mlflow.org/docs/latest/tracking.html#statsmodels)

- [Spark](https://mlflow.org/docs/latest/tracking.html#spark)

- [Fastai](https://mlflow.org/docs/latest/tracking.html#fastai)

- [Pytorch](https://mlflow.org/docs/latest/tracking.html#pytorch)

<br/>
<br/>
<br/>
<br/>
<br/>


#### MLflow Models

MLflow Models offer a convention for packaging machine learning models in multiple flavors, and a variety of tools to help you deploy them. 

Each Model is saved as a directory containing arbitrary files and a descriptor file that lists several “flavors” the model can be used in. 

For example, a TensorFlow model can be loaded as a TensorFlow DAG, or as a Python function to apply to input data. MLflow provides tools to deploy many common model types to diverse platforms: for example, any model supporting the “Python function” flavor can be deployed to a **Docker-based REST** server, to cloud platforms such as Azure ML and AWS SageMaker, and as a user-defined function in Apache Spark for batch and streaming inference. 

If you output MLflow Models using the Tracking API, MLflow also automatically remembers which Project and run they came from.

Each MLflow Model is a directory containing arbitrary files, together with an MLmodel file in the root of the directory that can define multiple flavors that the model can be viewed in.

This can be viewed with the UI: view model weights, files describing the model’s environment and dependencies, and sample code for making predictions with the model flavors : a convention that deployment tools can use to understand the model.

MLflow defines several “standard” flavors that all of its built-in deployment tools support, such as a “Python function” flavor that describes how to run the model as a Python function

Model signatures define input and output schemas for MLflow models, providing a standard interface to codify and enforce the correct use of your models; “What inputs does it expect?” and “What output does it produce?”

Example signature:
``` {r667, eval=F, echo=T}
  inputs: '[{"name": "sepal length (cm)", "type": "double"}, {"name": "sepal width
      (cm)", "type": "double"}, {"name": "petal length (cm)", "type": "double"}, {"name":
      "petal width (cm)", "type": "double"}, {"name": "class", "type": "string", "optional": "true"}]'
    outputs: '[{"type": "integer"}]'
```
    
You can pass this in as an object when you programmatically register your model:

Example flavors: 

- [Python Function (python_function)](https://mlflow.org/docs/latest/models.html#python-function-python-function)

- [R Function (crate)](https://mlflow.org/docs/latest/models.html#r-function-crate)

- [H2O (h2o)](https://mlflow.org/docs/latest/models.html#h2o-h2o)

- [Keras (keras)](https://mlflow.org/docs/latest/models.html#keras-keras)

- [MLeap (mleap)](https://mlflow.org/docs/latest/models.html#mleap-mleap)

- [PyTorch (pytorch)](https://mlflow.org/docs/latest/models.html#pytorch-pytorch)

- [Scikit-learn (sklearn)](https://mlflow.org/docs/latest/models.html#scikit-learn-sklearn)

- [Spark MLlib (spark)](https://mlflow.org/docs/latest/models.html#spark-mllib-spark)

- [TensorFlow (tensorflow)](https://mlflow.org/docs/latest/models.html#tensorflow-tensorflow)

- [ONNX (onnx)](https://mlflow.org/docs/latest/models.html#onnx-onnx)

- [MXNet Gluon (gluon)](https://mlflow.org/docs/latest/models.html#mxnet-gluon-gluon)

- [XGBoost (xgboost)](https://mlflow.org/docs/latest/models.html#xgboost-xgboost)

- [LightGBM (lightgbm)](https://mlflow.org/docs/latest/models.html#lightgbm-lightgbm)

- [CatBoost (catboost)](https://mlflow.org/docs/latest/models.html#catboost-catboost)

- [Spacy(spaCy)](https://mlflow.org/docs/latest/models.html#spacy-spacy)

- [Fastai(fastai)](https://mlflow.org/docs/latest/models.html#fastai-fastai)

- [Statsmodels (statsmodels)](https://mlflow.org/docs/latest/models.html#statsmodels-statsmodels)

- [Prophet (prophet)](https://mlflow.org/docs/latest/models.html#prophet-prophet)

- [Pmdarima (pmdarima)](https://mlflow.org/docs/latest/models.html#pmdarima-pmdarima)

- [OpenAI (openai) (Experimental)](https://mlflow.org/docs/latest/models.html#openai-openai-experimental)

- [LangChain (langchain) (Experimental)](https://mlflow.org/docs/latest/models.html#langchain-langchain-experimental)

- [John Snow Labs (johnsnowlabs) (Experimental)](https://mlflow.org/docs/latest/models.html#john-snow-labs-johnsnowlabs-experimental)

- [Diviner (diviner)](https://mlflow.org/docs/latest/models.html#diviner-diviner)

- [Transformers (transformers) (Experimental)](https://mlflow.org/docs/latest/models.html#transformers-transformers-experimental)

<br/>
<br/>
<br/>
<br/>
<br/>

#### MLflow Projects

MLflow Projects are a standard format for packaging reusable data science code. 

Each project is simply a directory with code or a Git repository, and uses a descriptor file or simply convention to specify its dependencies and how to run the code. For example, projects can contain a conda.yaml file for specifying a Python Conda environment. 

When you use the MLflow Tracking API in a Project, MLflow automatically remembers the project version (for example, Git commit) and any parameters. You can easily run existing MLflow Projects from GitHub or your own Git repository, and chain them into multi-step workflows.

An MLflow project is a way to organize and package your machine learning code, dependencies, and parameters in a reproducible manner. It provides a standardized format for defining and running your machine learning experiments, making it easier to reproduce and share your work.

An MLflow project consists of the following components:

1. Code: Your machine learning code is the core of the project. It can be written in any language or framework that MLflow supports, such as Python, R, or Scala. The code typically includes data preprocessing, model training, evaluation, and any other relevant steps.

2. Dependencies: MLflow allows you to specify the dependencies required for your project to run. This includes any libraries, packages, or external resources that your code relies on. By specifying dependencies, you ensure that the project can be executed in a consistent and isolated environment.

3. Parameters: MLflow projects allow you to define parameters that can be easily configured and tuned during the execution of the project. Parameters can be numerical values, strings, or other types, and they enable you to customize your experiments without modifying the code.

4. Entry Points: An entry point is a specific function or script within your project that serves as the starting point for execution. MLflow projects can define multiple entry points, allowing you to run different parts of your code or explore various experiment configurations.

By organizing your code, dependencies, parameters, and entry points within an MLflow project, you can easily package and share your work with others. It provides a standardized way to run your code, ensuring consistent execution across different environments and making it easier for collaborators to reproduce your experiments.

MLflow projects are language-agnostic, meaning you can use MLflow with various programming languages and frameworks. MLflow provides a command-line interface (CLI) and an API that allow you to run and manage your projects locally or in distributed computing environments such as Databricks.

Overall, MLflow projects simplify the process of organizing, running, and sharing your machine learning code, making it a convenient tool for managing the end-to-end machine learning lifecycle.


##### MLflow Projects in DataBricks


```{r, eval=F, echo=T}
mlflow run
```

<br/>
<br/>
<br/>
<br/>
<br/>

#### MLflow Registry

MLflow Registry offers a centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of an MLflow Model. 

It provides model lineage (which MLflow experiment and run produced the model), model versioning, stage transitions (for example from staging to production or archiving), and annotations.

##### Model 

An MLflow Model is created from an experiment or run that is logged with one of the model flavor’s ```mlflow.<model_flavor>.log_model()``` methods. Once logged, this model can then be registered with the Model Registry.

##### Registered Model 

An MLflow Model can be registered with the Model Registry. A registered model has a unique name, contains versions, associated transitional stages, model lineage, and other metadata.

##### Model Version 

Each registered model can have one or many versions. When a new model is added to the Model Registry, it is added as version 1. Each new model registered to the same model name increments the version number.

##### Model Stage 

Each distinct model version can be assigned one stage at any given time. MLflow provides predefined stages for common use-cases such as *Staging, Production or Archived*. You can transition a model version from one stage to another stage.

##### Annotations and Descriptions 

You can annotate the top-level model and each version individually using Markdown, including description and any relevant information useful for the team such as algorithm descriptions, dataset employed or methodology.

##### Model Alias

You can create an alias for a registered model that points to a specific model version. You can then use an alias to refer to a specific model version via a model URI or the model registry API. For example, you can create an alias named **Champion** that points to version 1 of a model named **MyModel**. You can then refer to version 1 of **MyModel** by using the URI ```models:/MyModel@Champion```.




```{r, eval=F, echo=T}
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

with mlflow.start_run() as run:
    X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    params = {"max_depth": 2, "random_state": 42}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Infer the model signature
    y_pred = model.predict(X_test)
    signature = infer_signature(X_test, y_pred)

    # Log parameters and metrics using the MLflow APIs
    mlflow.log_params(params)
    mlflow.log_metrics({"mse": mean_squared_error(y_test, y_pred)})

    # Log the sklearn model and register as version 1
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",
        signature=signature,
        registered_model_name="sk-learn-random-forest-reg-model",
    )
```

<br/>
<br/>
<br/>
<br/>
<br/>

