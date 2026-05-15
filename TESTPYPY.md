
## Uploading the package to (Test)PyPi

Here we will use the trusted publisher route to upload the package. [PyPi documentation](https://docs.pypi.org/trusted-publishers/)

## Configure a pending publisher

Full docs at https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/

- In the menu on the left hand side in (Test)PyPi, go to the "publishing" option
- Add a new pending publisher, giving the PyPi project name, github repo owner, repository name and workflow name (here publish.yml)
- Click add
- Check that project shows in the pending publishers list

## Add workflow to github actions

In `.github/workflows` there is a github actions script that
- Builds the package with uv
- Uploads it to (Test)PyPi using the `pypa/gh-action-pypi-publish` action

Currently this runs on pushes to the main branch, but this can be configured differently.
