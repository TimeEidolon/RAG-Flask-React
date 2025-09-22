import React from "react"
import {useInitial} from "./useInitial.js"

const ComponentPreviews = React.lazy(() => import("../../dev/previews.jsx"))

export {
    ComponentPreviews,
    useInitial
}