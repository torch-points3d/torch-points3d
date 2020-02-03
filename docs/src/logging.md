<h2> Custom logging </h2>

We use a custom hydra logging message which you can find wihtin ```/conf/hydra/job_logging/custom.yaml```

```yaml
hydra:
    job_logging:
        formatters:
            simple:
                format: "%(message)s"
        root:
            handlers: [debug_console_handler, file_handler]
        version: 1
        handlers:
            debug_console_handler:
                level: DEBUG
                formatter: simple
                class: logging.StreamHandler
                stream: ext://sys.stdout
            file_handler:
                level: DEBUG
                formatter: simple
                class: logging.FileHandler
                filename: train.log
        disable_existing_loggers: False
```