# Adding an ingress URL through the ngrok Agent SDK for Python

[ngrok](https://ngrok.com) is a globally distributed reverse proxy commonly used for quickly getting a public URL to a
service running inside a private network, such as on your local laptop. The ngrok agent is usually
deployed inside a private network and is used to communicate with the ngrok cloud service.

By default the authtoken in the NGROK_AUTHTOKEN environment variable will be used. Alternatively one may be specified in
the `settings.json` file, see the Examples below. Retrieve your authtoken on the [Auth Token page of your ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken), signing up is free.

# Documentation

For a list of all available options, see [the configuration documentation](https://ngrok.com/docs/ngrok-agent/config/) or [the connect example](https://github.com/ngrok/ngrok-py/blob/main/examples/ngrok-connect-full.py).

The ngrok Python SDK is [on github here](https://github.com/ngrok/ngrok-py). A quickstart guide and a full API reference are included in the [ngrok-py Python API documentation](https://ngrok.github.io/ngrok-py/).

# Running

To enable ngrok install the requirements and then add `--extension ngrok` to the command line options, for instance:

```bash
pip install -r extensions/ngrok/requirements.txt
python server.py --extension ngrok
```

In the output you should then see something like this:

```bash
INFO:Loading the extension "ngrok"...
INFO:Session created
INFO:Created tunnel "9d9d0944dc75ff9d3aae653e5eb29fe9" with url "https://d83706cf7be7.ngrok.app"
INFO:Tunnel "9d9d0944dc75ff9d3aae653e5eb29fe9" TCP forwarding to "localhost:7860"
INFO:Ingress established at https://d83706cf7be7.ngrok.app
```

You can now access the webui via the url shown, in this case `https://d83706cf7be7.ngrok.app`. It is recommended to add some authentication to the ingress, see below.

# Example Settings

In `settings.json` add a `ngrok` key with a dictionary of options, for instance:

To enable basic authentication:
```json
{
    "ngrok": {
        "basic_auth": "user:password"
    }
}
```

To enable OAUTH authentication:
```json
{
    "ngrok": {
        "oauth_provider": "google",
        "oauth_allow_domains": "asdf.com",
        "oauth_allow_emails": "asdf@asdf.com"
    }
}
```

To add an authtoken instead of using the NGROK_AUTHTOKEN environment variable:
```json
{
    "ngrok": {
        "authtoken": "<token>",
        "authtoken_from_env":false
    }
}
```