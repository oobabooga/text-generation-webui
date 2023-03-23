import os
from .i18n import get_i18n

cred_names: list[str] = ['api_id', 'api_hash', 'bot_token']
t = get_i18n()

def get_creds(path) -> dict[str, str]:
  credentials_path=f'{path}/credentials'

  if not os.path.isfile(credentials_path):
    raise Exception(t('error.credential_file.not_found', {'/path/': path}))

  creds: dict[str, str] = {}

  with open(credentials_path, 'r') as file:
    for line in file.readlines():

      line = line.split('=')
      line = list(map(lambda e: e.strip(), line))

      if line[0] in cred_names:
        creds[line[0]] = line[1]

  if len(creds.keys()) < 3:
    raise Exception(t('error.credential_file.is_invalid'))

  return creds
