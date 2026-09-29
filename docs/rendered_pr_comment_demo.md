# Rendered PR Comment Demo

This file is intentionally simple and text-heavy so rendered Markdown review comments are easy to test in a pull request.

## Why This Exists

The goal is to verify that inline review feedback can be anchored from rendered Markdown selections without switching to source view.

## Demo Checklist

- Open the PR "Files changed" tab.
- Switch the diff to rendered/rich mode when available.
- Select a sentence in this document and post a review comment.
- Confirm the comment appears on the expected changed line in GitHub.

## Notes

If the selected text appears multiple times, mapping may be ambiguous and you may need to pick the closest candidate line.
