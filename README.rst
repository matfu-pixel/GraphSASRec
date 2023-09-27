|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Распознавание эмоций персонажей в сцене фильма с помощью мультимодального трансформера
    :Тип научной работы: M1P
    :Автор: Артем Сергеевич Матвеев
    :Научный руководитель: кандидат физико-математических наук, Майсурадзе Арчил Ивериевич

Abstract
========

В современном мире кинематограф играет важную роль в формировании общественного мнения и культуры. Каждый фильм имеет свои особенности, которые могут вызвать разные эмоции у зрителей. В данной работе производится попытка автоматизации процесса определения эмоций персонажей в кино с помощью хорошо зарекомендовавшей себя архитектуры трансформер. Задача определения эмоций формулируется как задача многоклассовой классификации. В качестве данных для обучения и валидации используется датасет MovieGraphs. На вход трансформер принимает сразу несколько абстракций: сцену (задается набором кадров) и персонажей (задаются своими bounding-box-ми). Предложенный подход демонстрирует свою эффективность в сравнении со стандартными бейзлайнами.

Research publications
===============================
1. 

Presentations at conferences on the topic of research
================================================
1. 

Software modules developed as part of the study
======================================================
1. A python package *mylib* with all implementation `here <https://github.com/intsystems/ProjectTemplate/tree/master/src>`_.
2. A code with all experiment visualisation `here <https://github.comintsystems/ProjectTemplate/blob/master/code/main.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/ProjectTemplate/blob/master/code/main.ipynb>`_.
