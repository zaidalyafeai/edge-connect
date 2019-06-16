# Base64 encode an image and save the encoded version as an environment variable
# to use in a POST request to the modelf

BASE64_IMAGE=$(base64 -i input_image.png | xargs echo "data:image/jpeg;base64," | sed "s/ //" )

# Make a POST request to the /classify command, receiving
curl http://0.0.0.0:8000/fill \
   -X POST \
   -H "content-type: application/json" \
   -d "{ \"input_image\": \"${BASE64_IMAGE}\" }"
