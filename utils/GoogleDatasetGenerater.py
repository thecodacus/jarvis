from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

imagedirectory="fortnite"
keyword="fortnite gameplay images"
limit=200

arguments = {"chromedriver":"chromedriver","keywords":keyword,"limit":limit, "output_directory":"dataset", "image_directory":imagedirectory}   #creating list of arguments
response.download(arguments)   #passing the arguments to the function