## How have reporting on topics surrounding covid changed in 2022 as compared to 2021?

<div align="center">
  
| File | Description |
|---|---|
| [Main Notebook](https://github.com/SitwalaM/nlp-topic-modelling/blob/main/Topic_Modelling_Final_TeamB.ipynb) | Main Notebook submitted for Twist Challenge  |
| [Data Extraction Notebook](https://github.com/SitwalaM/nlp-topic-modelling/blob/main/scripts/nlp_dag.py) | Kaggle Notebook used for Data Extraction |
|[Dashboard](https://public.tableau.com/views/Tanamadosi1/Dashboard?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link)| Dashboard using Flask|
  
</div>

As global cases of COVID19 began to rise from Dec 2019, we witnessed an increase in news reporting on COVID19 related topics (such as panic, shortage, testing, quarantine etc) globally on a daily basis. Every news agency be it TV or Online had something to report, mostly negative. If 2020 was dominated by the news of how COVID-19 spread across the globe, then 2021 has so far been focused on ending the pandemic through vaccine distribution. As vacciness began to roll out and the rate of deaths reducing significantly, the rate of reporting began to decrease and media focus was shifted to new topics.

This project is to compare how reporting on topics surrounding covid has changed between 2021 and 2022. The initial plan was to compare the first quarter of 2021 and 2022 but given the size of the data, it had to be limited to the months of January only. After which a classification model will be built using NLP on the data from January 2022.

# Data

The data was extracted from the gdelt public data available on Google big query and because of the limitations and difficulty in getting it directly from Google Big Query, I extracted it through kaggle which can access Big Query directly. The final step was to upload it into google drive for use in Colab.

This data reflects news reported online. The notebook used has been uploaded in GitHub.

As I earlier stated in my introduction, the data had to be limited to just one month of each year due to the download size.

# EDA

The use of line graph, bar graph and pie chart was used on the raw data to visualize the change in reporting on covid related topics for 2021 and 2022. The output of the visualizations did prove the fact that reporting on covid related topics indeed reduced in January 2022 which could be as a result of the roll out of vaccines in 2020 hence less reported cases and deaths.


