# Household Material Classification
<div align="center"> 
<img src="https://github.com/brandelt4/libfreenect2_alex/blob/main/bin.png"/>
</div>

## Idea
In this project, I wanted to create a prototype of a household recycling machine. After researching the subject quite a bit, it became clear to me that household recycling is very likely to be automated in the future – it is both more convenient and less error-prone. However, the problem is extremely complex and will require a great team of engineers to complete successfully. 


## Design
For now, the mechanical design of the bin will remain a secret. Here, I am sharing some code and my thoughts on how the central component of such a bin could work – the material classification system. At the time of the project, no such system was created or open-source. Looking back, computer vision (perhaps, also with the use of depth image) is the way to go; however, the tech I used here can also be promising for this use. Original implementation was done by ken1row/libfreenect2_altered.

### Installation:
1. Install the necessary software required for libfreenect2 library (https://github.com/OpenKinect/libfreenect2).
2. Replace the contents of the /build folder with the contents of this repository.


