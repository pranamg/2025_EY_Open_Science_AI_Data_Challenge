# EY Open Science AI & Data Challenge Program

Aligned with the UN Sustainable Development Goals, the EY Open Science AI & Data Challenge is an annual competition that gives university students, early-career professionals, and EY people the opportunity to develop data models using artificial intelligence and computing technology to create open-source solutions that address critical climate issues, building a more sustainable future for society and the planet. 

Since the first challenge began in 2021, over 34,000 participants from 143 countries have signed on to tackle pressing issues such as preserving biodiversity and managing wildfires. Their skills, creativity and vision are helping to shape a more stable and sustainable future for society and the planet. 


## The 2025 EY Open Science AI and Data Challenge: Cooling Urban Heat Islands 

The EY Open Science AI & Data Challenge calls for innovators to address the Urban Heat Island effect using AI. Develop ML models to predict city temperatures and aid urban design for cooler, sustainable environments. Contribute to global efforts against climate change and enhance urban resilience. Join us in shaping a livable future for city dwellers

The 2025 EY Open Science AI and Data Challenge: Cooling Urban Heat Islands (EY Participants)
The EY Open Science AI & Data Challenge calls for innovators to address the Urban Heat Island effect using AI. Develop ML models to predict city temperatures and aid urban design for cooler, sustainable environments. Contribute to global efforts against climate change and enhance urban resilience. Join us in shaping a livable future for city dwellers.

## Overview 

### About the Challenge:
Aligned with the [United Nations Sustainable Development Goals](https://sdgs.un.org/goals) and the [EY Ripples program](https://www.ey.com/en_gl/about-us/corporate-responsibility), the EY Open Science AI & Data Challenge is an annual competition that gives university students, early-career professionals and EY people the opportunity to develop data models using artificial intelligence (AI) and computing technology to create solutions that address critical climate issues, building a more sustainable future for society and the planet.

The 2025 AI & data challenge is focused on a phenomenon known as the urban heat island effect, a situation that occurs due to the high density of buildings and lack of green space and water bodies in urban areas. Temperature variations between rural and urban environments can exceed 10-degrees Celsius in some cases and cause significant health-, social- and energy-related issues. Those particularly vulnerable to heat-related problems include young children, older adults, outdoor workers, and low-income populations.

All output from the challenge can help bring cooling relief to vulnerable communities, but entrants with top scores will take home cash prizes and receive an invitation to an exciting awards celebration.

### Problem Statement:
The goal of the challenge is to develop a machine learning model to predict heat island hotspots in an urban location. Additionally, the model should be designed to discern and highlight the key factors that contribute significantly to the development of these hotspots within city environments.

Participants will be given near-surface air temperature data in an index format, which was collected on 24 July 2021 using a ground traverse in the Bronx and Manhattan region of New York City. This dataset constitutes traverse points (latitude and longitude) and their corresponding UHI (Urban Heat Island) index values. Participants will use this dataset to build a regression model to predict UHI Index values for a given set of locations.

It is important to understand that the UHI Index at any given location is indicative of the relative temperature difference at that specific point when compared to the city's average temperature. This index serves as a crucial metric for assessing the intensity of heat within different urban zones.

The challenge begins on January 20, 2025 and will end on March 20, 2025.

Check the FAQs tab for more details and good luck on the challenge!

## FAQs 

### Challenge related questions

#### Is there a cost to join a challenge? 
No, there is no cost associated with challenge participation.

#### What will challenge participants do? 
The competition provides entrants with an opportunity to use AI for good and take part in helping build a more sustainable future for society and the planet.
Participants will be asked to develop machine learning and AI models using ground temperature data, satellite data, building footprints and heights, and weather data to identify location and severity of urban hot spots. Those who proceed to finals will be asked to develop a practical "business plan" that describes how the model summitted could be applied by local beneficiaries to help address the issue of urban heat islands, including the use of expanded datasets and analysis.

#### How does the challenge process work? 
Once registered and the challenge begins (January 20, 2025), participants will be provided with many datasets to consider for their models. The ability to determine which datasets and parameters are the most important for model accuracy will determine the finalists. These datasets include:
- Ground datasets, temperature data collected by CAPA Strategies using ground traverses
- European Sentinel-2 Satellite – Spectral bands and indices to identify land classification and assess proximity to vegetation or water
- NASA Landsat Satellite - Land Surface Temperature data to assess surface responses from buildings and land
- Building Footprints and Height – Building locations can be used to assess urban density and building heights can be used to assess shading.
- Local Weather Data – Local weather data (temp, humidity, wind speed and direction) can be used in the model.

Data will be used to build machine learning models to forecast temperatures at micro-scales (meters) across a city. Models will be tested against known temperatures at specific locations to determine an overall accuracy (least-squares error). In addition, participants will be asked to submit a short document describing their analysis approach and address scaling such solutions to other cities, additional datasets that could improve model accuracy, socioeconomic impact, and practical application for local decision-makers.

#### Who will own any IP produced from the challenge? 
The participants own the solutions submitted to the challenge. The organizers encourage participants to share the winning solutions in a designated open-source repository to ensure lasting impact of the challenge toward resolving sustainability issues.

#### What specific skills or knowledge do I need to complete in the challenge, e.g. coding experience? 
We recommend participants to have a minimum level of coding experience. However, if this is not the case for you, any of the Python training modules should help you get through the challenge. Some past participants who had little coding experience submitted very solid models.

#### How do I download the datasets? 
The datasets and the sample Python notebook are available in the data/raw and docs folders.

#### Is it permissible to use other data sets? 
Yes, additional datasets can be used to build the model if they are free and publicly available.

#### What is the evaluation methodology for the challenge? 
An out-of-sample validation dataset will be provided to participants. Submissions/predictions (.csv file), will be compared with the ground truth file and a R-squared (R²) score will be generated to evaluate the performance of the model.

#### What specific skills or knowledge do I need to complete in the challenge? 
We recommend participants have a minimum level of coding experience. Knowledge in data science, machine learning and Python programming are beneficial to completing the challenge.

#### If I see that a participant achieved the highest score, can I still continue in the challenge? 
Yes, you should continue making submissions to the challenge, as evaluations are made using multiple criteria such as score, methodology, approach, innovation etc.

#### I am unable to find the GeoTIFF image that has been used in the sample model notebook. Will this be shared with participants? 
We do not plan to share the GeoTIFF image used in the sample model notebook with participants. You are encouraged to explore and download your own GeoTIFF image; additionally, we have provided a sample code to create a basic GeoTIFF file.

#### Can I use any model I want? Or do I have to edit/extend the model provided within the sample notebook? 
The sample model notebook is intended as a reference to demonstrate a workflow for model development. You are not required to follow the algorithm/model used in the notebook. You are free to use any regression algorithm/model of your choice to build the solution. Ensure that the algorithm/model you choose is open-source and that you clearly document your process in the code.

#### What platforms or software can participants use? 
Participants can use any software or platform to build their model. The recommended development language is Python. A compute resource of 4 core 32 GB memory is sufficient for the challenge.

#### Which software do I need to install to build the model? 
Participants are required to construct the model utilizing the Python programming language. To visualize the images, you may install open-source applications like QGIS and then add in a raster layer to view the GeoTIFF file. If you choose to build the model on a local environment, you may need to set up an integrated development environment to run the code and might need to add extra Python libraries for model development.

## Data Description
### Target Dataset:
Near-surface air temperature data in an index format was collected on 24 July 2021 across the Bronx and Manhattan regions of New York City in the United States. The data was collected in the afternoon between 3:00 pm and 4:00 pm. This dataset includes time stamps, traverse points (latitude and longitude) and the corresponding Urban Heat Island (UHI) Index values for 11229 data points. These UHI Index values are the target parameters for your model.

Please find the dataset [here](/data/raw/Training_data_uhi_index.csv).

### Feature Datasets:
Participants can leverage many datasets to consider for their models. Their ability to analyze which datasets and parameters are the most important for model development will determine the model performance. The following are the recommended satellite datasets:
- [European Sentinel-2 optical satellite data](/notebook/Sentinel2_GeoTIFF.ipynb)
- [NASA Landsat optical satellite data](/notebook/Landsat_LST.ipynb)

These datasets can be extracted from Microsoft Planetary Computer Portal's data catalog. Please see the sample notebooks for more details.

### Additional Datasets:
Participants can also explore the following datasets in their model development journey:

- [Building footprints](/data/raw/Building_Footprint.kml) of the Bronx and Manhattan regions
- [Detailed local weather dataset](/data/raw/NY_Mesonet_Weather.xlsx) of the Bronx and Manhattan regions on 24 July 2021

Additionally, participants are allowed to use additional datasets for their models, provided those datasets are open and available to all public users and the source of such datasets are referenced in the model.

### Validation Dataset:
After building the machine learning model, you need to predict the UHI index values on the locations identified in the [validation dataset](/data/raw/Submission_template.csv). Predictions on the validation dataset need to be saved in a CSV file and uploaded to the challenge platform to get a score on the ranking board.

### Supporting Material:
Participants can refer to the following material before starting model development:
- [Participants' guidance document](/docs/2025%20EY%20Open%20Science%20AI%20Data%20Challenge%20Participant%20Guidance.pdf), which provides a detailed overview of urban heat island concepts, relevant datasets, and suggestions for model development
- [Jupyter notebook](/notebook/UHI%20Experiment%20Sample%20Benchmark%20Notebook%20V5.ipynb) where a sample model has been built by using challenge training data
- [Sample notebook](/notebook/Sentinel2_GeoTIFF.ipynb) to download a GeoTIFF image from the Sentinel-2 satellite dataset


Terms of Use and Licensing requirements for the datasets:

### Training Data:
- Description: Ground temperature data over New York City on July 24, 2021 (CSV format)
- Contributors: Climate, Adaptation, Planning, Analytics (CAPA) Strategies
- Data Host: Center for Open Science - [https://www.cos.io](https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fwww.cos.io%2F&amp;data=05%7C02%7CSaurabh.Agarwal4%40gds.ey.com%7C9244dd5c059b44ffc08e08dd2f46bd98%7C5b973f9977df4bebb27daa0c70b8482c%7C0%7C0%7C638718703073133499%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&amp;sdata=zczBas9UBkYJL6dPyjtQS6GeBUP5vrB%2BKSorgNj%2BU3Q%3D&amp;reserved=0)
- Terms of Use: [https://github.com/CenterForOpenScience/cos.io/blob/master/TERMS_OF_USE.md](https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fgithub.com%2FCenterForOpenScience%2Fcos.io%2Fblob%2Fmaster%2FTERMS_OF_USE.md&amp;data=05%7C02%7CSaurabh.Agarwal4%40gds.ey.com%7C9244dd5c059b44ffc08e08dd2f46bd98%7C5b973f9977df4bebb27daa0c70b8482c%7C0%7C0%7C638718703073154040%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&amp;sdata=wsLOr6ZTt%2Bq5F%2B4WA1aZCAnKMb3PTxkW8GYrgS0Vn1o%3D&amp;reserved=0)
- License: Apache 2.0 &gt; [https://github.com/CenterForOpenScience/cos.io/blob/master/LICENSE](https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fgithub.com%2FCenterForOpenScience%2Fcos.io%2Fblob%2Fmaster%2FLICENSE&amp;data=05%7C02%7CSaurabh.Agarwal4%40gds.ey.com%7C9244dd5c059b44ffc08e08dd2f46bd98%7C5b973f9977df4bebb27daa0c70b8482c%7C0%7C0%7C638718703073167085%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&amp;sdata=GVuuZsijQr%2BTWs8JZTcYbL50aR%2FNcOsWBEblVykeltU%3D&amp;reserved=0)

### Satellite Data (Sentinel-2 Sample Output)
- Description: Copernicus Sentinel-2 sample data from 2021 obtained from the Microsoft Planetary Computer (TIFF format)
- Contributors: European Space Agency (ESA), Microsoft
- Data Host: Microsoft Planetary Computer - [https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a](https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fplanetarycomputer.microsoft.com%2Fdataset%2Fsentinel-2-l2a&amp;data=05%7C02%7CSaurabh.Agarwal4%40gds.ey.com%7C9244dd5c059b44ffc08e08dd2f46bd98%7C5b973f9977df4bebb27daa0c70b8482c%7C0%7C0%7C638718703073180142%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&amp;sdata=jkn8yQQQzYjagNwCcSA25MaXzgxn%2BBqrmmKq3u2w%2BOE%3D&amp;reserved=0)
- Terms of Use: [https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice](https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fsentinel.esa.int%2Fdocuments%2F247904%2F690755%2FSentinel_Data_Legal_Notice&amp;data=05%7C02%7CSaurabh.Agarwal4%40gds.ey.com%7C9244dd5c059b44ffc08e08dd2f46bd98%7C5b973f9977df4bebb27daa0c70b8482c%7C0%7C0%7C638718703073192982%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&amp;sdata=80QPsRYJJjSpuBpZdxNpXYaIWowT0vNMWLqgFR845JQ%3D&amp;reserved=0)
- License: [https://creativecommons.org/licenses/by-sa/3.0/igo/](https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fcreativecommons.org%2Flicenses%2Fby-sa%2F3.0%2Figo%2F&amp;data=05%7C02%7CSaurabh.Agarwal4%40gds.ey.com%7C9244dd5c059b44ffc08e08dd2f46bd98%7C5b973f9977df4bebb27daa0c70b8482c%7C0%7C0%7C638718703073205553%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&amp;sdata=zRfzR%2B%2BX0CYa2HpL1FCqi%2FwgwrHU3UcSWJm4vo3M63E%3D&amp;reserved=0)

### Building Footprint Data
- Description: Building footprint polygons over the data challenge region of interest (KML format)
- Contributors: Open Data Team at the NYC Office of Technology and Innovation (OTI) - New York City Open Data Project
- Data Host: [https://data.cityofnewyork.us/Housing-Development/Building-Footprints/nqwf-w8eh](https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fdata.cityofnewyork.us%2FHousing-Development%2FBuilding-Footprints%2Fnqwf-w8eh&amp;data=05%7C02%7CSaurabh.Agarwal4%40gds.ey.com%7C9244dd5c059b44ffc08e08dd2f46bd98%7C5b973f9977df4bebb27daa0c70b8482c%7C0%7C0%7C638718703073219192%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&amp;sdata=PrD1JzHTjzKDXPRmYMAyBI20JqUjb75tu8OhXP5gQds%3D&amp;reserved=0)
- Terms of Use: [https://www.nyc.gov/html/data/terms.html](https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fwww.nyc.gov%2Fhtml%2Fdata%2Fterms.html&amp;data=05%7C02%7CSaurabh.Agarwal4%40gds.ey.com%7C9244dd5c059b44ffc08e08dd2f46bd98%7C5b973f9977df4bebb27daa0c70b8482c%7C0%7C0%7C638718703073231949%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&amp;sdata=vvGk52If4WT4CfGYxCDZ3Enn8unkXaeBIBDSSCJYAHw%3D&amp;reserved=0) and [https://www.nyc.gov/home/terms-of-use.page](https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fwww.nyc.gov%2Fhome%2Fterms-of-use.page&amp;data=05%7C02%7CSaurabh.Agarwal4%40gds.ey.com%7C9244dd5c059b44ffc08e08dd2f46bd98%7C5b973f9977df4bebb27daa0c70b8482c%7C0%7C0%7C638718703073243671%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&amp;sdata=uWJJSniost1cnjY%2FODCISv5I%2F55FlntiRnZy8K4DyXE%3D&amp;reserved=0)
- License: [https://github.com/CityOfNewYork/nyc-geo-metadata#Apache-2.0-1-ov-file](https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fgithub.com%2FCityOfNewYork%2Fnyc-geo-metadata%23Apache-2.0-1-ov-file&amp;data=05%7C02%7CSaurabh.Agarwal4%40gds.ey.com%7C9244dd5c059b44ffc08e08dd2f46bd98%7C5b973f9977df4bebb27daa0c70b8482c%7C0%7C0%7C638718703073255234%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&amp;sdata=Kk%2BmrP1Z1QDcVcv5m9DiPYzbQS4d2BT%2FKpjREFZBdLc%3D&amp;reserved=0)

### Weather Data
- Description: Detailed weather data collected every 5 minutes at two locations (Bronx and Manhattan). Includes surface air temperature (2-meters), relative humidity, average wind speed, wind direction, and solar flux.
- Contributors: Contributors: New York State Mesonet
- Data Host: [https://nysmesonet.org/](https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fnysmesonet.org%2F&amp;data=05%7C02%7CSaurabh.Agarwal4%40gds.ey.com%7C9244dd5c059b44ffc08e08dd2f46bd98%7C5b973f9977df4bebb27daa0c70b8482c%7C0%7C0%7C638718703073266796%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&amp;sdata=gWS2R7P1MuyJ06WwfSTPT0Pplr%2F8S80rsazjMit2NoE%3D&amp;reserved=0)
- Terms of Use: [https://nysmesonet.org/about/data](https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fnysmesonet.org%2Fabout%2Fdata&amp;data=05%7C02%7CSaurabh.Agarwal4%40gds.ey.com%7C9244dd5c059b44ffc08e08dd2f46bd98%7C5b973f9977df4bebb27daa0c70b8482c%7C0%7C0%7C638718703073279815%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&amp;sdata=RtQfvlVeA0x5JczZyzJmrw%2BS9wydOERllcO5XJn49Nw%3D&amp;reserved=0)
- License: [https://nysmesonet.org/documents/NYS_Mesonet_Data_Access_Policy.pdf](https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fnysmesonet.org%2Fdocuments%2FNYS_Mesonet_Data_Access_Policy.pdf&amp;data=05%7C02%7CSaurabh.Agarwal4%40gds.ey.com%7C9244dd5c059b44ffc08e08dd2f46bd98%7C5b973f9977df4bebb27daa0c70b8482c%7C0%7C0%7C638718703073291303%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&amp;sdata=8JFISW%2FkfY37vW51jfmhTW%2FfOwwbpKB%2F2rD0e7rHI40%3D&amp;reserved=0)