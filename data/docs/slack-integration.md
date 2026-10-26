# Slack integration

The Slack integration brings Fernhill links, tasks and updates into Slack. It is available on every plan. Free workspaces can connect one Slack workspace, and paid plans can connect up to 5. Slack Enterprise Grid is supported on the Fernhill Enterprise plan.

## Connecting Slack

An admin opens Settings > Integrations > Slack and clicks Connect, then approves the requested permissions in Slack. Each person then runs `/fernhill connect` in Slack once to link their own Fernhill account.

## Link previews

When someone pastes a Fernhill link in Slack, the integration shows a preview with the page title, the first lines of content and the last editor. Previews only appear for people who linked their Fernhill account and who can access the page, so sharing a link never exposes a private page.

## Creating tasks from Slack

There are two ways to turn a conversation into a task:

- Hover over a message, open the More actions menu and choose Create Fernhill task. The message text becomes the task description, with a link back to the Slack thread.
- Type `/fernhill task` followed by a title, for example `/fernhill task Update the onboarding checklist`.

New tasks go to the default space chosen in the integration settings.

## Slash commands

- `/fernhill search <query>` searches pages you can access and shows the top 5 results only to you.
- `/fernhill subscribe <space-key>` posts page updates from a space to the current channel. Updates are grouped into a digest every 15 minutes to keep channels quiet.
- `/fernhill unsubscribe <space-key>` stops those updates.

Task assignments and mentions can also be sent to you as Slack direct messages. Turn them on in Settings > Notifications.

## Data and disconnecting

Fernhill stores Slack user and channel IDs and the text of messages you turn into tasks. It does not read any other messages. Disconnecting in Settings > Integrations > Slack revokes the tokens immediately.
