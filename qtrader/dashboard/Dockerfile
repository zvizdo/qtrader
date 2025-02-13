FROM python:3.8.15

EXPOSE 8080

COPY . ./

RUN /usr/local/bin/python -m pip install --upgrade pip

RUN pip3 install streamlit==1.22.0
RUN pip3 install plotly==5.14.1
RUN pip3 install prefect==1.*
RUN pip3 install google-cloud-storage==2.8.0

WORKDIR .

CMD streamlit run ./qtrader/dashboard/main.py --server.port=8080 --server.address=0.0.0.0
# gcloud builds submit --region=us-central1 --config ./qtrader/dashboard/cloudbuild.yaml
# gcloud run deploy qtrader-dashboard --region=us-central1 --image=gcr.io/as-dev-anze/qtrader-dashboard:latest