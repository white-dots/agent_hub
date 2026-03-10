# Impact-Aware Development

## Before Editing Files
When modifying any source file, call the `impact_check` MCP tool first with the file path.
If risk is HIGH, preserve the exported interface — do not rename functions, change signatures,
or remove classes without also updating all listed dependents.

## After Editing Files
Call `affected_tests` with all changed file paths. Run the suggested test command.
Fix any failures before moving on to the next task.

## Frontend Changes
After modifying any .tsx, .jsx, .vue, .svelte, or .css file, use `preview_screenshot`
to verify the rendered output. Use `preview_inspect` on the changed component to verify
styles and layout. Do not assume CSS changes are correct without visual verification.
