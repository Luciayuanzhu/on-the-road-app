# Backend Codex Window Prompt

Use this prompt in the backend Codex window.

## Working Directory Rule

You may only create, edit, or delete files inside:

- `/Users/luciad/hackathon proj/backend`

You may read project docs from the parent directory for reference, but do not modify files outside the backend folder.

## Task

Build the Railway backend for the live AI companion MVP.

## Required References

Read these files first:

- `/Users/luciad/hackathon proj/README.md`
- `/Users/luciad/hackathon proj/BACKEND_TASKS.md`
- `/Users/luciad/hackathon proj/FRONTEND_BACKEND_CONTRACT.md`

## Product Requirements

- Resolve anonymous identity by mapping `deviceId` one-to-one to `userId`
- Create a new `userId` when a device has no mapping
- Own all prompt construction and settings injection
- Sessions start with mic off on the client, so backend must support sessions with no audio input
- Camera assist is explicit and session-scoped through `visual_assist.start` and `visual_assist.stop`
- Bookmark summaries are asynchronous and bookmarks support editable notes
- Contract stability matters more than inventing extra fields

## Execution Rules

- Work only inside `/Users/luciad/hackathon proj/backend`
- Keep implementation aligned with the contract document
- Do not push prompt text construction to the frontend
- Build the smallest stable backend that supports end-to-end MVP behavior
