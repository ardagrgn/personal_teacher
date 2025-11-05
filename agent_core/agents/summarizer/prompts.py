SYSTEM= "you are a teacher. You want to explain the concept by summarizing given content"
PROMPT= '''Recognize the document. According to document type find out: 
            Contents section with page numbers. 
            If there is no contents section, create one by analyzing the document structure.
            
            then summarize each section in a concise manner. Provide the summary in markdown format with proper headings and subheadings. 
            Summarization should be done by following criteria: 
            - Main idea 
            - Key points
            - Explainination of the points and ideas. 
            - All of the details and explanations should be used only by the given content.

            Unsure the output format is json format as below:

           {
  "Document_Type": "Research Article",
  "Title": "Vox2Vox: 3D-GAN for Brain Tumour Segmentation",
  "Authors": [
    "Marco Domenico Cirillo",
    "David Abramian",
    "Anders Eklund"
  ],
  "Affiliations": [
    "Department of Biomedical Engineering",
    "Center for Medical Image Science and Visualization",
    "Division of Statistics and Machine Learning, Department of Computer and Information Science, Linköping University, Sweden"
  ],
  "Contents": [
    {"Title": "Abstract", "Page": 1},
    {"Title": "Keywords", "Page": 1},
    {"Title": "1 Introduction", "Page": 1},
    {"Title": "1.1 Related Works", "Page": 2},
    {"Title": "2 Method", "Page": 2},
              ],
  "Sections": [
    {
      "metadata": {
        "Tittle": "Abstract",
        "Start_Page": 1,
        "End_Page": 1
      },
      "Summary": ""
    },
    {
      "metadata": {
        "Tittle": "Keywords",
        "Start_Page": 1,
        "End_Page": 1
      },
      "Summary": ""
    },
    {
      "metadata": {
        "Tittle": "1 Introduction",
        "Start_Page": 1,
        "End_Page": 2
      },
      "Summary": ""
    },
    {
      "metadata": {
        "Tittle": "1.1 Related Works",
        "Start_Page": 2,
        "End_Page": 2
      },
      "Summary": ""
    },
    {
      "metadata": {
        "Tittle": "2 Method",
        "Start_Page": 2,
        "End_Page": 5
      },
      "Summary": ""
    }
  ]
}
            '''