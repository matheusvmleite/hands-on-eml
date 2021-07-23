# Packging your application


### Build your docker image

```
sudo docker build -t xgb-rest-predictor .
```

### Run your docker image

```
sudo docker run -d --name xgb -p 80:80 xgb-rest-predictor 
```

### Check your container logs

```
sudo docker logs xgb 
```

### Test your service

```
curl -X 'POST' \
  'http://localhost/predictions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "instance": [
    4.7, 4.0, 0.6, 0.6
  ]
}'
```