# Graph Perturbation Synthetic

## Design
Step 1: Input Descriptions
Provide descriptions of BP, BM and problem statement

Input the Business Process descriptions, Business Meaning description, and the Problem Statement that defines what we're trying to analyze or solve.

Step 2: Determine Domain and Context
Identify the domain and context of the data

Analyze the input descriptions to determine the specific domain (e.g., finance, supply chain, healthcare) and the contextual purpose for which the data will be used. Context is for what is the data being used in the domain. Ex. For Pharma Domain- Drug Trial 1 can be context.

Step 3: Gather Attribute Information
Collect attribute information from BEs

Extract attribute data and metadata from the Business Entities that are part of the graph structure.

Step 4: Establish Context of Attributes
Define the context in which attributes will be used

Combine the domain/context information from Step 2 with the BE attributes and the BPs steps identify the dependency of the entity attributes and seperate them into inputs and outputs. 

Step 5: Generate Data Constraints
Create constraints for the data based on context

Using the contextual understanding of attributes, define constraints that govern valid data ranges, relationships, dependencies, and business rules. This can be variance, standard deviation, or distributions to which the data belongs to. Use the sample data given for this. If the ranges or distrubution of data is provided then we can use that.

Step 6: Use ATSD or Data Generation Library
Employ ATSD or a library to generate data

To generate the data ATSD has been used. Based on the results from above and the entity types from the BEs payload is created to call ATSD for synthetic data generation. As there is no sample data or ranges available it will produce random data as of now.


## Working

To run the python file you can use the below command:

```python
python test.py 'https://cdn-new.gov-cloud.ai/_ENC(nIw4FQRwLOQd0b8T2HcImBUJ5a9zjZEImv/UhJi8/+yUl7Ez+m0qAiCCaOJbNgi5)/onnx/3857e5e6-3173-4675-991f-e9d3994c1580_$$_V1_attribute.json' both 100 token.txt
```

For the API run the app.py and make the below calls:

Endpoint: `POST /api/graph_perturbation_synthetic`

Request Body:

```json
{
  "cdn_url": "https://cdn-new.gov-cloud.ai/path/to/file.json",
  "mode": "both",
  "row_count": 100,
  "model_name": "gpt-oss:20b"
}
```
Response:

```json
{
    "job_id": "ad6eb23a-0264-4e0e-87ae-2ed0a1059ac3",
    "status": "queued",
    "message": "Job queued successfully. Use GET /api/job/{job_id} to check status.",
    "created_at": "2026-02-06T13:23:25.857570"
}
```

To check status of the above:

Endpoint: `GET /api/job/{job_id}`


Response:

```json
{
    "job_id": "ad6eb23a-0264-4e0e-87ae-2ed0a1059ac3",
    "status": "completed",
    "message": "Synthetic data generation completed successfully",
    "created_at": "2026-02-06T07:47:23.354849",
    "completed_at": "2026-02-06T07:49:05.271144",
    "result": {
        "total_business_meanings": 2,
        "results": [
            {
                "business_meaning_id": "bm_c648b75b81d0",
                "business_meaning_name": "Accelerate AI Deployment",
                "entities": [
                    {
                        "entity_id": "be_589001fc2726",
                        "entity_name": "Business Problem Statement",
                        "cdn_url": "https://cdn.gov-cloud.ai/_ENC(4+j2JOgE1QQdq6yO427Uztql2TlqlMwKUOg5QJcVQ5XUgB/GP4/J5WLrrqWMDU3q)/spde_data/87f101cf-1700-4c55-b029-86f91aa0932a/61f2c629-cf73-4a31-a961-64f2eba3b186_$$_V1_business_problem_statement.csv",
                        "attribute_count": 8
                    },
                    {
                        "entity_id": "be_83825da33d61",
                        "entity_name": "Audit Log",
                        "cdn_url": "https://cdn.gov-cloud.ai/_ENC(4+j2JOgE1QQdq6yO427Uztql2TlqlMwKUOg5QJcVQ5XUgB/GP4/J5WLrrqWMDU3q)/spde_data/fc5fbeb0-8047-4639-acc0-efedc3687164/ed172023-e982-4219-a4a6-fabbece0e5f4_$$_V1_audit_log.csv",
                        "attribute_count": 8
                    },
                    {
                        "entity_id": "be_5b26ad3edfb7",
                        "entity_name": "Performance Metrics",
                        "cdn_url": "https://cdn.gov-cloud.ai/_ENC(4+j2JOgE1QQdq6yO427Uztql2TlqlMwKUOg5QJcVQ5XUgB/GP4/J5WLrrqWMDU3q)/spde_data/9c8aa684-8928-43f2-8ca9-09dcfeac6932/99731fa9-fade-49ea-ac87-f409dfd538d5_$$_V1_performance_metrics.csv",
                        "attribute_count": 8
                    }
                ]
            }
        ]
    },
    "error": null
}
```

