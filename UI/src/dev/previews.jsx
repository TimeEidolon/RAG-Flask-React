import React from 'react'
import {ComponentPreview, Previews} from '@react-buddy/ide-toolbox'
import {PaletteTree} from './palette'
import {ChatForIndex} from "../components/Chat/ChatForIndex.jsx";

const ComponentPreviews = () => {
    return (
        <Previews palette={<PaletteTree/>}>
            <ComponentPreview path="/ChatForIndex">
                <ChatForIndex/>
            </ComponentPreview>
        </Previews>
    )
}

export default ComponentPreviews