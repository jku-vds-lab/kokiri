## Release notes

*List of addressed issues and PRs since the last release*


## Checklists

### Release preparation

* [x] Create new `release-x.x.x` branch (based on `develop` branch)
* [ ] Collect changes and write [release notes](#release-notes)
* [ ] Draft release PR in GitHub that merges the `release-x.x.x` into the `master` branch

### Release dependencies first

In case of dependent Phovea/TDP repositories follow [dependency tree](https://wiki.datavisyn.io/phovea/fundamentals/development-process#dependency-hierarchy) from the top:

* [ ] Release dependent repositories if they contain changes first before proceeding here
* [ ] Replace git dependencies in *package.json* with new version range (e.g., `"tdp_core": "^2.3.1"` when published on npm **or** `"tdp_core": "github:datavisyn/tdp_core#semver:^8.0.0"` for non-published repositories)
* [ ] Replace git dependencies in *requirements.txt* with new version range (e.g., `phovea_server>=2.3.0,<3.0.0` when published on pipy **or** `-e git+https://github.com/datavisyn/tdp_core.git@v8.0.0#egg=tdp_core` for non-published repositories)
* [ ] Commit and push new dependencies
* [ ] Wait until build is successful
* [ ] Repeat with other repositories/dependencies or proceed with next section

### Update version

* [ ] Check version numbers of dependencies again
* [ ] Check if build is successful
* [ ] Update this version number following [semver](https://semver.org)
* [ ] Commit and push *package.json* with new version number
* [ ] Assign reviewer and wait for final review
* [ ] Merge this pull request into master branch
* [ ] Add release label (i.e., `release: major`, `release: minor`, or `release: patch`)

### Publish pip release

The steps of this section are only necessary if the code is public and should be published to the pypi registry.

* [ ] `chmod -R o+w .` in the cloned repository directory (to provide write access to the CircleCI Linux user)
* [ ] `rm -rf dist && rm -rf build`
* [ ] `docker run -it -v $(pwd):/phovea circleci/python:3.7-buster-node-browsers /bin/bash` and continue inside the container
* [ ] `cd /phovea`
* [ ] `sudo pip install -r requirements.txt && sudo pip install -r requirements_dev.txt && sudo pip install twine`
* [ ] `npm run dist`
* [ ] Ensure only two files are in the *dist* directory (*.whl and *.tar.gz)
* [ ] Ensure that both files contain the new version number
* [ ] `twine upload --repository-url https://upload.pypi.org/legacy/ dist/*`
* [ ] Login with `caleydo-bot`
* [ ] Check release on [pipy.org](https://pypi.org/)

### Create GitHub release

* [ ] Draft a new release (Code -> Releases -> Draft a new release)
* [ ] Use new version number as tag (e.g., `v2.3.1`)
* [ ] Copy [release notes](#release-notes)
* [ ] Publish release

### Prepeare next develop release

* [ ] Switch to `develop` branch
* [ ] Merge `master` branch into `develop` (`git merge origin/master`)
* [ ] Update version in *package.json* to `<next patch version>-SNAPSHOT` (e.g., `2.3.1` to `2.3.2-SNAPSHOT`)
* [ ] Revert dependencies in *requirements.txt* to develop branches (e.g., `-e git+https://github.com/phovea/phovea_server.git@develop#egg=phovea_server`)
* [ ] Commit and push changes
 
### 🏁 Finish line

* [ ] Inform colleagues and customers about the release
* [ ] Celebrate the new release 🥳
