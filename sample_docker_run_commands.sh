./instabrain -s subjectid -c configname

docker run --name container_name -v $PWD/config.yml:/usr/src/app/config.yml -v path/to/local/ref:path/to/folder/ref image_name subjectid

# need to move config to config.yml.sample
# gitignore actual config.yml

# docker build -t imagename .
