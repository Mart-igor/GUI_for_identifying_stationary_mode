
# ППО идентификации стационарных режимов 

[![GitHub Stars](https://img.shields.io/github/stars/Mart-igor/GUI_for_identifying_stationary_mode.svg)](https://github.com/Mart-igor/GUI_for_identifying_stationary_mode/stargazers)
[![GitHub commits](https://img.shields.io/github/commits-since/aregtech/areg-sdk/v1.0.0.svg?style=social)](https://GitHub.com/aregtech/areg-sdk/commit/)
[![Watchers](https://img.shields.io/github/watchers/aregtech/areg-sdk?style=social)](https://github.com/aregtech/areg-sdk/watchers)


*Автоматизация эффективна, так как она сокращает время выполнения задач, минимизирует ошибки и позволяет сосредоточиться на стратегически важных аспектах работы.*

---

## Introduction

В настоящее время в различных отраслях активно используют математические модели для управления технологическими процессами. Чтобы правильно определить параметры этих моделей, лучше всего работать в условиях стационарного режима, однако с учетом многосвязности объектов управления и большого числа регулируемых параметров специалистам решать задачу поиска стационарных режимов без помощи алгоритмов затруднительно.

Данное ППО позволяет решить задачу автоматического поиска стационарных рабочих режимов.


![Chat Preview](gif_and_screen\Clipchamp4-ezgif.com-video-to-gif-converter.gif)

---

## Table of content
 - [Tech Stack](#tech-stack)
 - [Demo](#demo)
 - [Documentation](#documentation)
 - [Usage](#usage)
 - [Deployment](#deployment)
 - [Feedback](#feedback)


<a id="tech-stack"><h2>Tech Stack</h2></a>

**Client:** React, Redux, TailwindCSS

**Server:** Node, Express

- **Язык программирования**: Python
- **Графический интерфейс**: PySide6
- **Многопоточность**: threading (стандартная библиотека Python)
- **База данных**: SQLite (или укажите другую СУБД, если используется)
- **Машинное обучение**: scikit-learn, pandas, numpy (или конкретные библиотеки, которые вы используете)
- **Стилизация интерфейса**: CSS (встроенный в PySide6)
- **Система контроля версий**: Git, GitHub

## Demo

Программа предоставляет удобный графический интерфейс для ввода данных и использует алгоритм оптимизации для их обработки.

![Демонстрация работы](screenshots/demo.gif)

В разделе "Info" приведена инструкция по использованию.

![Демонстрация работы](screenshots/demo.gif)


## Documantation {#documentation}

Для более детального изучения алгоритма и его математического обоснования, вы можете ознакомиться с моей статьей по данной теме:  
- [Документация и теоретические основы](https://linktodocumentation)

 
## Usage {#usage}

### Основные шаги для использования программы:

1. **Загрузка данных**:
   - Загрузите CSV-файл с данными через интерфейс программы.

2. **Отображение данных**:
   - После загрузки файла данные автоматически отобразятся в виде таблицы в интерфейсе.

3. **Выбор параметров**:
   - В выпадающих списках выберите необходимые параметры для анализа.
   - Постройте график изменения выбранного параметра.

4. **Анализ стационарных режимов**:
   - Перейдите на следующую страницу интерфейса.
   - Укажите начало (`x_min`) и конец (`x_max`) участка данных, на котором будет проводиться оценка стационарных режимов. Рекомендуется выбирать диапазон от 3000 до 10000 значений.

5. **Результаты анализа**:
   - Программа выведет таблицу с результатами:
     - **"stationary"** — оценка стационарности.
     - **"assessment"** — ваша личная оценка.
   - Постройте график, на котором вы сможете вручную выделить стационарные участки. Эти данные также будут записаны в таблицу.

6. **Оптимизация**:
   - Нажмите кнопку **"Optimize"**, чтобы программа рассчитала оптимальные весовые коэффициенты.
   - Результатом будет метрика классификации **F1-score**.

7. **Визуализация и экспорт данных**:
   - После завершения анализа вы можете:
     - Построить график за весь период, на котором будут отмечены стационарные и нестационарные участки.
     - Выгрузить размеченные данные в файл для дальнейшего использования.


## Deployment {#deployment}

To deploy this project run

```bash
  npm run deploy
```


## Feedback {#feedback}

Feel free to send us feedback on Twitter or file an issue. Feature requests are always welcome. If you wish to contribute, please take a quick look at the guidelines!

## Show your support

Please ⭐️ this repository if you find this project usefull!

And you can support this project here --> 
<a href="https://www.patreon.com/FranckAbgrall">
  <img src="https://c5.patreon.com/external/logo/become_a_patron_button@2x.png" width="160">
</a>