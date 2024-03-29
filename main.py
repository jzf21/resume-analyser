from typing import Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi import FastAPI
import docx2txt 


app = FastAPI()


@app.get("/")
def read_root():
    
    resume=docx2txt.process("Resume-NonMEC.docx")
    job_description=docx2txt.process("job_description.docx")
    text=[resume,job_description]
    cv=CountVectorizer()
    count_matrix=cv.fit_transform(text)
    print(cosine_similarity(count_matrix))
    match_percentage=cosine_similarity(count_matrix)[0][1]*100
    return {"message": resume,match_percentage:match_percentage}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
@app.post("/match/")
async def match_resume_to_job(job_description_file: UploadFile = File(...), resume_file: UploadFile = File(...)):
    try:
        # Read contents of uploaded files
        job_description = docx2txt.process(job_description_file.file)
        resume = docx2txt.process(resume_file.file)
        
        # Calculate cosine similarity
        text = [resume, job_description]
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(text)
        similarity_matrix = cosine_similarity(count_matrix)
        match_percentage = similarity_matrix[0][1] * 100
        
        return JSONResponse({"message": "Files processed successfully.", "match_percentage": match_percentage})
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)