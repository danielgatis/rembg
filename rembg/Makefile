# keep help at the beginning - this will be used as default command
.description: Generate list of targets with descriptions
help:
	@grep -oz '^.description: .*\n.*:' Makefile | sed 'N;s;\.description: \(.*\)\n\(.*\):;"\x1b[1\;32m make \2 \x1b[0m" "\1";' | xargs printf "%-50s - %s\n"

# MACROS

DOCKER = docker-compose -f docker-compose.development.yml
DOCKER_DEV = $(DOCKER)
CFLAGS='-Wno-warning'

# use the rest as arguments for "run"
ARGS = `arg="$(filter-out $@,$(MAKECMDGOALS))" && echo $${arg:-${1}}`

# COMMANDS

# common
.description: start all services in DEVELOPMENT environment
start:
	make stop backend
	$(DOCKER_DEV) --profile all up -d

.description: stop all app services. You can pass service name as an argument
stop:
	$(DOCKER) --profile all stop $(call ARGS)

.description: stop all services
down:
	$(DOCKER) --profile all down

.description: build docker images
build:
	$(DOCKER_DEV) --profile all build

.description: docker logs with follow option. You can pass service name as argument.
flogs:
	$(DOCKER) logs --follow $(call ARGS)

.description: docker logs. You can pass service name as argument.
logs:
	$(DOCKER) logs $(call ARGS)

.description: exec in the running service (pass service name and command as an argument). `make exec redis redis-cli`
exec:
	$(DOCKER) exec $(call ARGS)

.description: attaches to backend process stdin
attach:
	docker attach --detach-keys=ctrl-c $$($(DOCKER_DEV) ps -q backend)

.description: launches bash in backend container
bash:
	$(DOCKER_DEV) exec backend bash

# Ignore unknown targets
%:
	@:

# ...and turn them into do-nothing targets
_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
$(eval $(_ARGS):dummy;@:)
