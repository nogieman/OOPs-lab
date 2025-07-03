#!/bin/bash

# Exit on error
#set -e

# Check if the VERSION.txt file exists
if [[ ! -f VERSION.txt ]]; then
    echo "Error: VERSION.txt does not exist."
    exit 1
fi

# Read the version from the file
VERSION=$(cat VERSION.txt)

# Validate the version format (semver)
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: VERSION.txt does not contain a valid semver (e.g., 1.2.3)."
    exit 1
fi

# Parse the version into major, minor, and patch
IFS='.' read -r MAJOR MINOR PATCH <<< "$VERSION"

# Ensure the user provided the field to update
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 [major|minor|patch]"
    exit 1
fi

FIELD=$1

# Update the specified field
case $FIELD in
    major)
        ((MAJOR++))
        MINOR=0
        PATCH=0
        ;;
    minor)
        ((MINOR++))
        PATCH=0
        ;;
    patch)
        ((PATCH++))
        ;;
    *)
        echo "Error: Invalid field '$FIELD'. Use 'major', 'minor', or 'patch'."
        exit 1
        ;;
esac

# Construct the new version
NEW_VERSION="$MAJOR.$MINOR.$PATCH"

# Write the new version back to the file
echo "$NEW_VERSION" > VERSION.txt

# Output the updated version
echo "Version updated to: $NEW_VERSION"


cat > src/version.h << EOF
#pragma once

#define GATICC_VERSION "$NEW_VERSION"
#define GATICC_BOOST_VERSION "$(awk -F " " '/constant BOOST_VERSION/{print $4}' third_party/boost/Jamroot)"
#define GATICC_PROTOBUF_VERSION "$(awk -F '["]' '/protoc_version/{print $4}' third_party/protobuf/version.json)"
EOF

git add VERSION.txt src/version.h
git commit -m "bump version to: $NEW_VERSION

Commit log:


# Describe this release
# - change X
# - move Y
# - adds support for Z
# - remove W

$(git log --oneline --no-decorate --no-merges v$VERSION..HEAD)

" -e
git tag "v$NEW_VERSION"
